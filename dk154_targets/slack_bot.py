
import logging
import os
from pathlib import Path


logger = logging.getLogger(__name__.split(".")[-1])

# Import WebClient from Python SDK (github.com/slackapi/python-slack-sdk)
try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
except ImportError as e:
    print(e)
    WebClient = None



class SlackManager:

    def __init__(self, token, channels):

        self.token = token
        self.client = WebClient(token = token)

        self.channels = channels


    @classmethod
    def from_config(cls, slack_config: dict=None):
        if WebClient is None:
            return None
        if slack_config is None:
            return None
        token = slack_config.get("token", None)
        if token is None:
            logger.info("no `token` in slack_config")
        
        channels = slack_config.get("channels", None)
        return cls(token, channels)


    def update_channels(self, text=None, files=None):
        if text is None and files is None:
            logger.info("no text or files...")

        if text is not None and files is None:
            for channel in self.channels:
                try:
                    result = self.client.chat_postMessage(
                        channel=channel,
                        text=text
                    )
                except SlackApiError as e:
                    logger.error(e)

        else:
            if isinstance(files, (str, Path)):
                files = [files]
            for fpath in files:
                for channel in self.channels:
                    try:
                        result = self.client.files_upload_v2(
                            channel=channel,
                            initial_comment=text,
                            file=str(fpath)
                        )
                    except SlackApiError as e:
                        logger.error(e)