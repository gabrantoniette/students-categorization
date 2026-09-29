from src.ingest import load_contract, to_bronze, to_silver, to_gold

if __name__ == "__main__":
    contract = load_contract()
    bronze = to_bronze(contract)
    silver, rejected = to_silver(contract)
    xs, ys = to_gold(contract)

    print(f"bronze  {len(bronze)} rows")
    print(f"silver  {len(silver)} valid | {len(rejected)} rejected | {len(bronze) - len(silver) - len(rejected)} empty or duplicate")
    print(f"gold    xs {xs.shape} | ys {ys.shape}")