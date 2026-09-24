from huggingface_hub.inference._providers._common import BaseConversationalTask


class NvidiaConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider="nvidia", base_url="https://integrate.api.nvidia.com")
