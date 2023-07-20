import subprocess
import pkg_resources
import json

def get_installed_distributions():
    return [d for d in pkg_resources.working_set]

def remove_package(package):
    subprocess.check_call(["pip", "uninstall", "-y", package])

def install_package(package):
    subprocess.check_call(["pip", "install", package])

def test_code():
    exist_status = subprocess.call(["python", "test_script.py"])
    if exist_status != 0:
        raise Exception
    
def install_requirement(init_packages):
    for package in init_packages:
        install_package(package)

def remove_requirement(remove_packages):
    for package in remove_packages:
        remove_package(package)

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--install', type=bool, default=False, help='install init packages')
    parser.add_argument('--remove', type=bool, default=False, help='remove unnecessary packages')
    parser.add_argument('--check', type=bool, default=False, help='check dependency packages')
    args = parser.parse_args()

    if args.check == True:
        packages = get_installed_distributions()

        necessary_packages = []
        unnecessary_packages = []
        for package in packages:
            if package.project_name == "pip":
                continue
            elif package.project_name == "torch":
                continue
            
            try:
                print(f"Removing {package}")
                remove_package(package.project_name)

                test_code()
                install_package(package.project_name)
                unnecessary_packages.append(package.project_name)
            
            except:
                print(f"Failed to test code without {package.project_name}")
                necessary_packages.append(package.project_name)
                install_package(package.project_name)
                
        print(necessary_packages)
        print(unnecessary_packages)
        data = {
            "necessary_packages": necessary_packages,
            "unnecessary_packages": unnecessary_packages
        }

        with open('dependency_check.json', mode='w', encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    elif args.install == True:
        init_packages = ["torch", "torchvision", "opencv-python", "kornia", "timm"]
        install_requirement(init_packages)
    
    elif args.remove == True:
        with open("dependency_check.json", mode="r", encoding="utf-8") as f:
            file = json.load(f)
        unnecessary_packages = file["unnecessary_packages"]
        remove_requirement(unnecessary_packages)

    else:
        print("Not valid options")
        raise Exception
    