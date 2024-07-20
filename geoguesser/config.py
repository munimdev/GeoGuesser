from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    google_api_key: str = ""


settings = Settings()
