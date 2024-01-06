import pathlib
RUN_LOCAL = False
ENCODER_PATH = "saved_models/316/decoder.h5"
DECODER_PATH = "saved_models/316/decoder.h5"
if RUN_LOCAL:
  ENCODER_PATH = r"saved_models/encoder.h5"
  DECODER_PATH = r"saved_models/decoder.h5"
  DECODER_PATH = r"C:\Users\joeli\Dropbox\Code\HM_2023\TrainedModels\no_duplicates_75_2_mask\decoder.h5"
  ENCODER_PATH = r"C:\Users\joeli\Dropbox\Code\HM_2023\TrainedModels\no_duplicates_75_2_mask\encoder.h5"
  ENCODER_PATH = r"C:\Users\joeli\Dropbox\AE_MC\saved_ml_models\SmallBatchSize\encoder_19_05.h5"
  DECODER_PATH = r"C:\Users\joeli\Dropbox\AE_MC\saved_ml_models\SmallBatchSize\decoder_19_05.h5"
else:
  ENCODER_PATH = "/content/Modify_Texture_Docker/saved_models/316/encoder.h5"
  DECODER_PATH = "/content/Modify_Texture_Docker/saved_models/316/decoder.h5"
  ENCODER_PATH = "/content/Modify_Texture_Docker/saved_models/no_duplicates_75_2_mask/encoder.h5"
  DECODER_PATH = "/content/Modify_Texture_Docker/saved_models/no_duplicates_75_2_mask/decoder.h5"
  ENCODER_PATH = "saved_models/316/decoder.h5"
  DECODER_PATH = "saved_models/316/decoder.h5"
  ENCODER_PATH = "saved_models/no_duplicates_75_2_mask/encoder.h5"
  DECODER_PATH = "saved_models/no_duplicates_75_2_mask/decoder.h5"