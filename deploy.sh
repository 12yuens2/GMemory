for file in template/deploy-templates/*.json; do
    az deployment group create --resource-group intrinsic-memory --template-file template/template.json --parameters $file

    echo "Deploy $file"

    sleep 1
done
