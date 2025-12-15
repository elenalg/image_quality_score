#!/usr/bin/env python3
"""
CLI to compute quality scores for images using ImageQuality.
Supports single images or directories and can export results to JSON.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

try:
    from .image_quality import ImageQuality
except ImportError:
    from image_quality import ImageQuality

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Get quality score(s) for images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python get_image_score.py image.jpg

  # Export to JSON
  python get_image_score.py image.jpg --output scores.json

  # Directory (non-recursive)
  python get_image_score.py --d /path/to/images

  # Directory recursive with custom extensions
  python get_image_score.py -d /path/to/images --recursive --extensions .jpg .png

  # Export to JSON
  python get_image_score.py image.jpg --output scores.json
        """
    )
    parser.add_argument("images", nargs="*", help="Image file paths")
    parser.add_argument(
        "--directory", "-d", type=str, help="Directory containing images"
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Search recursively in directory"
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"],
        help="Image extensions to include (default: jpg jpeg png bmp tiff tif)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="scores.json",
        help="Write results to JSON file (default: scores.json)"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not write JSON output even if an output path is set"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    image_paths = []

    # Add provided image paths
    for img in args.images:
        p = Path(img)
        if p.exists():
            image_paths.append(p)
        else:
            logger.warning("Image does not exist: %s", img)

    # Add from directory
    if args.directory:
        dir_path = Path(args.directory)
        try:
            temp_paths = ImageQuality.collect_image_paths(
                dir_path, extensions=args.extensions, recursive=args.recursive
            )
            image_paths.extend(temp_paths)
        except ValueError as e:
            logger.error(e)
            return 1

    # Deduplicate paths
    image_paths = sorted(set(image_paths))

    if not image_paths:
        logger.error("No images to process. Provide paths or a valid directory.")
        return 1

    analyzer = ImageQuality(use_iqa=False)

    results = []
    errors = 0
    for img_path in image_paths:
        try:
            score = analyzer.get_quality_score(img_path)
            results.append({
                "image": str(img_path),
                **score
            })
        except Exception as e:
            errors += 1
            logger.error("Error processing %s: %s", img_path, e)
            results.append({
                "image": str(img_path),
                "error": str(e)
            })

    summary = {
        "total": len(image_paths),
        "errors": errors,
        "processed": len(results) - errors
    }

    output = {
        "results": results,
        "summary": summary
    }

    # Print concise output
    print("\n=== Image Quality Scores ===")
    print(f"Total: {summary['total']}  Processed: {summary['processed']}  Errors: {summary['errors']}")
    for res in results:
        print(f"\nImage: {res['image']}")
        if 'error' in res:
            print(f"  Error: {res['error']}")
            continue
        print(f"  Quality: {res['quality_score']:.3f}")
        print(f"    Sharpness: {res['sharpness_score']:.3f} (raw: {res['metrics']['sharpness']:.1f})")
        print(f"    Exposure:  {res['exposure_score']:.3f} (raw: {res['metrics']['exposure']:.3f})")
        # BRISQUE disabled in this script

    # Save JSON if requested
    if not args.no_save:
        out_path = Path(args.output) if args.output else Path("scores.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info("Results saved to %s", out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
