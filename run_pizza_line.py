import os
import autotwin_gmglib as gmg

# config_path = os.path.join("pizza-line-v3", "config.json")
config_path = os.path.join("pizza-line-v4", "config.json")
config = gmg.load_config(config_path)
gmg.import_log(config)
gmg.import_knowledge(config)
log = gmg.load_log(config)
model = gmg.generate_model(log, config)
gmg.save_model(model, config)
model = gmg.load_model(config)
gmg.show_model(model)
model_id = gmg.export_model(model, log, config)
print("ID of generated model in SKG:")
print(model_id)
