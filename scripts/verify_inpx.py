with open("outputs/vissim/highway_network.inpx", encoding="utf-8") as f:
    content = f.read()

print(content[:3000])
print(f"\nTotal file size: {len(content)} characters")