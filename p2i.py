from pdf2image import convert_from_path
from PIL import Image
import argparse


def pdf_to_long_image(pdf_path, out_path, direction):
    # 1) PDF 转多张 PIL.Image
    pages = convert_from_path(pdf_path)

    # ------------ 竖向拼接（长图） ------------
    if direction == "vertical":
        widths = [p.width for p in pages]
        heights = [p.height for p in pages]

        total_height = sum(heights)
        max_width = max(widths)

        # 生成空白大图
        long_img = Image.new("RGB", (max_width, total_height), color=(255, 255, 255))

        # 逐页贴进去
        y_offset = 0
        for p in pages:
            long_img.paste(p, (0, y_offset))
            y_offset += p.height

    # ------------ 横向拼接（横长图） ------------
    elif direction == "horizontal":
        widths = [p.width for p in pages]
        heights = [p.height for p in pages]

        total_width = sum(widths)
        max_height = max(heights)

        long_img = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))

        x_offset = 0
        for p in pages:
            long_img.paste(p, (x_offset, 0))
            x_offset += p.width

    else:
        raise ValueError("direction must be 'vertical' or 'horizontal'")

    long_img.save(out_path)
    return long_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--p",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--o",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--d",
        type=str,
        default="vertical",
        choices=["vertical", "horizontal"],
    )
    args = parser.parse_args()

    pdf_to_long_image(args.p, out_path=args.o, direction=args.d)


if __name__ == "__main__":
    main()
    print("✅ done")
