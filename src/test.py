from src.models.AutoEncoder.AE import AutoEncoder
from src.utils.model_stats.stats import save_model_specs
from src.utils.tools.tools import ROOT_PATH

if __name__ == "__main__":
    # datapath = "../data/raw/frappe/"
    ae = AutoEncoder([22,9])
    save_model_specs(ae, ROOT_PATH)

    # df_frappe = PreProcessDataNCFContextual(datapath)
