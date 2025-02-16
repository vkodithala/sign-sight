import pickle

# Specify the pickle file name.
pickle_file = "collected_asl_data.pkl"

# Load the existing data.
with open(pickle_file, "rb") as f:
    data = pickle.load(f)

# Print available keys.
print("Available classes in the data:")
for key in data.keys():
    print(key, end="  ")
print("\n")

# Prompt user for the letter to delete.
letter = (
    input("Enter the letter (or 'space') for which you want to delete the data: ")
    .strip()
    .lower()
)

# Confirm deletion.
confirm = (
    input(f"Are you sure you want to delete all data for '{letter}'? (y/n): ")
    .strip()
    .lower()
)
if confirm == "y":
    # Option 1: Remove the key entirely:
    # del data[letter]
    # Option 2: Clear the list for that key:
    data[letter] = []
    print(f"Data for '{letter}' has been deleted.")
else:
    print("No changes made.")

# Save the updated data back to the pickle file.
with open(pickle_file, "wb") as f:
    pickle.dump(data, f)

print(f"Updated data has been saved to {pickle_file}.")
