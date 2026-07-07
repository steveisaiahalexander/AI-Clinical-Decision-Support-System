from src.core.config import Config

print("PROJECT_ROOT:", Config.PROJECT_ROOT)
print("DATA_DIR:", Config.DATA_DIR)
print("RAW_DATA_DIR:", Config.RAW_DATA_DIR)
print("MODEL_DIR:", Config.MODEL_DIR)
print("OUTPUT_DIR:", Config.OUTPUT_DIR)
print("API_DIR:", Config.API_DIR)
print("APP_DIR:", Config.APP_DIR)

Config.initialize()