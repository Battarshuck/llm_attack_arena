import argparse
from prompt_process import load_json
import attack
import defence
import download_models
from utils import model_names_list

def run_attack(model_name, attack_type):
    """
    Execute an attack on a specified model using the chosen attack type.
    """
    if attack_type == "AutoDAN":
        print(f"Applying AutoDAN attack to {model_name}")
        attack_instance = attack.AutoDAN(model=model_name)
        attack_instance.run()
    elif attack_type == "GPTFuzz":
        print(f"Applying GPTFuzz attack to {model_name}")
        attack_instance = attack.GPTFuzz(model=model_name)
        attack_instance.run()
    elif attack_type == "DeepInception":
        print(f"Applying DeepInception attack to {model_name}")
        attack_instance = attack.DeepInception(model=model_name)
        attack_instance.run()
    elif attack_type == "Tap":
        print(f"Applying Tap attack to {model_name}")
        attack_instance = attack.Tap(model=model_name)
        attack_instance.run()
    elif attack_type == "Pair":
        print (f"Applying Pair attack to {model_name}")
        attack_instance = attack.Pair(model=model_name)
        attack_instance.run()
    elif attack_type == "Jailbroken":
        print(f"Applying Jailbroken attack to {model_name}")
        attack_instance = attack.Jailbroken(model=model_name)
        attack_instance.run()
    elif attack_type == "TemplateJailbreak":
        print (f"Applying TemplateJailbreak attack to {model_name}")
        attack_instance = attack.TemplateJailbreak(model=model_name)
        attack_instance.run()
    elif attack_type == "Parameters":
        print(f"Applying Parameters attack to {model_name}")
        attack_instance = attack.Parameters(model=model_name)
        attack_instance.run()
    elif attack_type == "GCG":
        print(f"Applying GCG attack to {model_name}")
        attack_instance = attack.GCG(model=model_name)
        attack_instance.run()
    else:
        print("Attack type not recognized.")

def main():
    parser = argparse.ArgumentParser(description="Run attack and defense mechanisms on AI models")
    parser.add_argument('--model', choices=model_names_list.keys(), required=False, help='Model to attack or defend')
    parser.add_argument('--mode', choices=['attack','process'], required=False, help='Whether to run an attack or apply a defense or process the results.')
    parser.add_argument('--type', required=False, help='Type of attack to run')
    parser.add_argument('--need-download', required=False,default="false", help='do you need to download the model?')
    args = parser.parse_args()

    if "false" in args.need_download:
        args.need_download = False
    else:
        args.need_download = True

    if args.need_download:
        print("downloading the model")            
        download_models.download(args.model)

    if args.mode == 'attack':
        run_attack(args.model, args.type)
    elif args.mode == 'process':
        ##NOTE This must happen before the defense is applied
        load_json(f'./Results/{args.model}')

if __name__ == "__main__":
    main()
