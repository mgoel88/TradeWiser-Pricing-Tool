"""
Create a sample rice image for testing.
"""

from PIL import Image, ImageDraw
import os
import random

def create_rice_sample():
    """Create a sample rice image."""
    # Create directory if it doesn't exist
    os.makedirs("assets/samples", exist_ok=True)
    
    # Create a new image with white background
    img = Image.new('RGB', (400, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw rice grains
    rice_color = (245, 245, 240)
    
    # Draw multiple grains
    for i in range(80):
        x = 30 + (i % 10) * 35
        y = 30 + (i // 10) * 30
        
        # Add a bit of randomness to position
        x += random.randint(-5, 5)
        y += random.randint(-5, 5)
        
        # Draw a grain (small oval)
        draw.ellipse((x, y, x + 15, y + 8), fill=rice_color, outline=(220, 220, 215))
    
    # Add some broken grains
    for i in range(15):
        x = 50 + random.randint(0, 300)
        y = 50 + random.randint(0, 200)
        
        # Draw a smaller broken grain
        draw.ellipse((x, y, x + 8, y + 5), fill=rice_color, outline=(220, 220, 215))
    
    # Add some slight color variations
    for i in range(10):
        x = 60 + random.randint(0, 300)
        y = 60 + random.randint(0, 200)
        
        # Draw a slightly yellowish grain
        yellow_tint = (250, 245, 220)
        draw.ellipse((x, y, x + 15, y + 8), fill=yellow_tint, outline=(225, 220, 200))
    
    # Save the image
    output_path = "assets/samples/rice_sample.jpg"
    img.save(output_path)
    print(f"Created rice sample at {output_path}")
    return output_path

if __name__ == "__main__":
    create_rice_sample()