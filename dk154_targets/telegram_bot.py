import logging
from pathlib import Path

try:
    import telegram
except:
    telegram = None

try:
    import tg_logger
except:
    tg_logger = None

logger = logging.getLogger(__name__.split(".")[-1])


class TelegramManager:

    def __init__(self, http_api, sudoers, users, verbose=False, linked_loggers=None):
        self.bot = telegram.Bot(token=http_api)
        self.sudoers = set(sudoers)
        all_users = set(users)
        self.users = all_users - self.sudoers # avoid duplicate sudoers.


        linked_loggers = linked_loggers or []
        if tg_logger is not None and verbose:
            for l_log in linked_loggers:
                tg_logger.setup(l_log, token=http_api, users=sudoers)
        

    @classmethod
    def from_config(cls, telegram_config, linked_loggers=None):
        if telegram is None:
            return None
        if telegram_config is None:
            return None
        http_api = telegram_config.get("http_api", None)
        if http_api is None:
            logger.info("no http_api in telegram config")
            return None
        sudoers = telegram_config.get("sudoers", [])
        if len(sudoers) == 0:
            logger.warning("no sudoers")
        users = telegram_config.get("users", [])
        
        verbose = telegram_config.get("verbose", False)

        manager = cls(http_api, sudoers, users, verbose=verbose)
        return manager

       
    def update_all_users(self, texts=None, img_paths=None, send_album=False, sudoers_only=False):
        user_ids = self.sudoers if sudoers_only else self.sudoers.union(self.users)
        texts = texts or []
        if isinstance(texts, str):
            texts = [texts]
        img_paths = img_paths or []
        if isinstance(img_paths, (str, Path)):
            img_paths = [img_paths]

        img_list = [
            telegram.InputMediaPhoto(open(Path(img_path), "rb")) for img_path in img_paths
        ]
        
        for user_id in user_ids:
            try:
                for text in texts:
                    self.bot.send_message(chat_id=user_id, text=text)

                if send_album:
                    self.bot.send_media_group(chat_id=user_id, media=img_list)
                else:
                    for img_path in img_paths:
                        with open(img_path, "rb") as img:
                            self.bot.send_photo(chat_id=user_id, photo=img)

            except telegram.error.TimedOut as e:
                logger.error(f"{user_id}: {e}")

                    