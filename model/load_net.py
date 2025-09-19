from model.IreseekNet import IreeSeek_cov_model
def get_model(MODEL_NAME, net_params):
    models = {
        #'IreeSeek_model': IreeSeek_model,
        #'IreeSeek': IreeSeek_model,
        'IreeSeek_conv': IreeSeek_cov_model,
        #'IreeSeek_bilstm_model':IreeSeek_bilstm_model
    }
        
    return models[MODEL_NAME](net_params)