from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GEOGUESSER_", env_nested_delimiter="__")

    google_api_key: str = ""


settings = Settings()
