from etl.silver.lendingclub import transform as transform_lendingclub
from etl.silver.transforms import ALL_TRANSFORMS


def main():
    for transform_cls in ALL_TRANSFORMS:
        transform_cls().run()
    transform_lendingclub()


if __name__ == "__main__":
    main()
