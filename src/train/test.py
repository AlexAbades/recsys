from src.utils.model_stats.stats import save_accuracy
check_point_path = "./src/checkpoints/nfc"

if __name__ == "__main__":
  save_accuracy(check_point_path + f"/best_epoch_{5}", a=1, b=2)