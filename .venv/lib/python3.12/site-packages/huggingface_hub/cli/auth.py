# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains commands to authenticate to the Hugging Face Hub and interact with your repositories.

Usage:
    # login and save token locally.
    hf auth login --token=hf_*** --add-to-git-credential

    # switch between tokens
    hf auth switch

    # list all tokens
    hf auth list

    # logout from all tokens
    hf auth logout

    # check which account you are logged in as
    hf auth whoami
"""

from typing import Annotated

import typer

from huggingface_hub.constants import ENDPOINT
from huggingface_hub.hf_api import whoami

from .._login import auth_list, auth_switch, login, logout
from ..utils import get_stored_tokens, get_token, logging
from ._cli_utils import FormatWithAutoOpt, OutputFormatWithAuto, TokenOpt, typer_factory
from ._output import out


logger = logging.get_logger(__name__)


auth_cli = typer_factory(help="Manage authentication (login, logout, etc.).")


@auth_cli.command(
    "login",
    examples=[
        "hf auth login",
        "hf auth login --token $HF_TOKEN",
        "hf auth login --token $HF_TOKEN --add-to-git-credential",
        "hf auth login --force",
    ],
)
def auth_login(
    token: TokenOpt = None,
    add_to_git_credential: Annotated[
        bool,
        typer.Option(
            help="Save to git credential helper. Useful only if you plan to run git commands directly.",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            help="Force re-login even if already logged in.",
        ),
    ] = False,
) -> None:
    """Login using a token from huggingface.co/settings/tokens."""
    login(token=token, add_to_git_credential=add_to_git_credential, skip_if_logged_in=not force)


@auth_cli.command(
    "logout",
    examples=["hf auth logout", "hf auth logout --token-name my-token"],
)
def auth_logout(
    token_name: Annotated[
        str | None,
        typer.Option(help="Name of token to logout"),
    ] = None,
) -> None:
    """Logout from a specific token."""
    logout(token_name=token_name)


def _select_token_name() -> str | None:
    token_names = list(get_stored_tokens().keys())

    if not token_names:
        logger.error("No stored tokens found. Please login first.")
        return None

    print("Available stored tokens:")
    for i, token_name in enumerate(token_names, 1):
        print(f"{i}. {token_name}")
    while True:
        try:
            choice = input("Enter the number of the token to switch to (or 'q' to quit): ")
            if choice.lower() == "q":
                return None
            index = int(choice) - 1
            if 0 <= index < len(token_names):
                return token_names[index]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")


@auth_cli.command(
    "switch",
    examples=["hf auth switch", "hf auth switch --token-name my-token"],
)
def auth_switch_cmd(
    token_name: Annotated[
        str | None,
        typer.Option(
            help="Name of the token to switch to",
        ),
    ] = None,
    add_to_git_credential: Annotated[
        bool,
        typer.Option(
            help="Save to git credential helper. Useful only if you plan to run git commands directly.",
        ),
    ] = False,
) -> None:
    """Switch between access tokens."""
    if token_name is None:
        token_name = _select_token_name()
    if token_name is None:
        print("No token name provided. Aborting.")
        raise typer.Exit()
    auth_switch(token_name, add_to_git_credential=add_to_git_credential)


@auth_cli.command("list | ls", examples=["hf auth list"])
def auth_list_cmd() -> None:
    """List all stored access tokens."""
    auth_list()


@auth_cli.command("token", examples=["hf auth token", "hf auth token | xargs curl -H 'Authorization: Bearer {}'"])
def auth_token() -> None:
    """Print the current access token to stdout."""
    token = get_token()
    if token is None:
        out.error("Not logged in. Run `hf auth login` first.")
        raise typer.Exit(code=1)
    print(token)
    out.hint("Run `hf auth whoami` to see which account this token belongs to.")


@auth_cli.command("whoami", examples=["hf auth whoami", "hf auth whoami --format json"])
def auth_whoami(
    format: FormatWithAutoOpt = OutputFormatWithAuto.auto,
) -> None:
    """Find out which huggingface.co account you are logged in as."""

    token = get_token()
    if token is None:
        out.error("Not logged in")
        raise typer.Exit(code=1)

    info = whoami(token)
    orgs = ",".join(org["name"] for org in info["orgs"]) or None
    endpoint = ENDPOINT if ENDPOINT != "https://huggingface.co" else None
    out.result("Logged in", user=info["name"], orgs=orgs, endpoint=endpoint)
