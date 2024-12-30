#!/bin/bash

# Function to make API calls
make_api_call() {
    local url="$1"
    local api_key="$2"
    curl -s -X GET "$url" \
         -H "accept: application/json" \
         -H "api_key: $api_key"
}

# Prompt for API key and VM ID
read -p "Enter your API key: " API_KEY
read -p "Enter the Virtual Machine ID: " VM_ID

# Make API calls
org_response=$(make_api_call "https://infrahub-api.nexgencloud.com/v1/auth/organizations" "$API_KEY")
vm_response=$(make_api_call "https://infrahub-api.nexgencloud.com/v1/core/virtual-machines/$VM_ID/events" "$API_KEY")

# Check if API calls were successful
if [[ $(echo "$org_response" | jq -r '.status') != "true" ]]; then
    echo "Error retrieving organization info"
    exit 1
fi

if [[ $(echo "$vm_response" | jq -r '.status') != "true" ]]; then
    echo "Error retrieving VM events"
    exit 1
fi

# Find the creation event and creator ID
creator_id=$(echo "$vm_response" | jq -r '.instance_events[] | select(.reason == "InstanceCreationSuccess") | .user_id')

if [[ -z "$creator_id" ]]; then
    echo "No creation event found for this VM."
    exit 1
fi

# Find the creator's information
creator_info=$(echo "$org_response" | jq -r --arg id "$creator_id" '.organization.users[] | select(.id == ($id | tonumber))')

if [[ -z "$creator_info" ]]; then
    echo "Creator not found in the organization's user list."
    exit 1
fi

# Extract and display creator's name and email
creator_name=$(echo "$creator_info" | jq -r '.name')
creator_email=$(echo "$creator_info" | jq -r '.email')

echo "VM created by:"
echo "Name: $creator_name"
echo "Email: $creator_email"