for file in template/deploy-templates/*; do
    az deployment group create --resource-group intrinsic-memory --template-file template/template.json --parameters $file

    echo "Deploy $file"

    sleep 1
done
