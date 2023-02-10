.PHONY: quality style

# Check code formatting
quality:
	python3 utils/code_formatter.py --check_only

# Format code samples automatically and check is there are any problems left that need manual fixing
style:
	python3 utils/code_formatter.py
