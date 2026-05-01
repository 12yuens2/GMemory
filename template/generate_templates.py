import json
import os

from typing import Any, Dict

def generate_parameters(task, memory, seed, model_name, api_key, 
                        container_name: str = "g-memory-console",
                        image_name: str = "intrinsic.azurecr.io/g-memory:latest"):
    
    parameters = {
        "$schema": "https://schema.management.azure.com/schemas/2015-01-01/deploymentParameters.json#",
        "contentVersion": "1.0.0.0",
        "parameters": {
            "availabilityZones": {"value": []},
            "containerName": {"value": f"{task}-{memory}-{seed}-{model_name}"},
            "location": {"value": "westeurope"},
            "imageType": {"value": "Private"},
            "imageName": {"value": image_name},
            "osType": {"value": "Linux"},
            "numberCpuCores": {"value": "1"},
            "memory": {"value": "1.5"},
            "restartPolicy": {"value": "Never"},
            "sku": {"value": "Standard"},
            "imageRegistryLoginServer": {"value": "intrinsic.azurecr.io"},
            "imageUsername": {"value": "intrinsic"},
            "imagePassword": {"value": None},
            "environmentVariable0": {"value": task},
            "environmentVariable1": {"value": memory},
            "environmentVariable2": {"value": seed},
            "environmentVariable3": {"value": model_name},
            "environmentVariable4": {"value": "https://agent-scaling-us-east-resource.cognitiveservices.azure.com/openai/v1/"},
            "environmentVariable5": {"value": api_key},
            "ipAddressType": {"value": "Public"},
            "ports": {"value": [{"port": "80", "protocol": "TCP"}]},
            "workspaceRegion": {"value": "westeurope"},
            "workspaceSubId": {"value": "437ce2b6-c1d8-4df6-b067-fc9209c568e9"},
            "workspaceResourceGroupName": {"value": "Multi-policy"},
            "workspaceName": {"value": "multipolicytra8454645234"}
        }
    }

    output_file = f"deploy-templates/{task}-{memory}-{seed}-{model_name}.json"
    
    with open(output_file, 'w') as f:
        json.dump(parameters, f, indent=4)
    
    print(f"✓ {output_file} created successfully")

if __name__ == "__main__":
    tasks = ["fever", "pddl", "alfworld", "sciworld"]
    memories = ["voyager", "g-memory"]
    seeds = ["42"]
    model_names = ["o3-mini"]

    api_key = os.environ["OPENAI_API_KEY"]

    for task in tasks:
        for memory in memories:
            for seed in seeds:
                for model_name in model_names:
                    generate_parameters(task, memory, seed, model_name, api_key)

