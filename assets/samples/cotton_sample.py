"""
Create a sample cotton image for testing.
"""

from PIL import Image, ImageDraw
import os
import random
import math

def create_cotton_sample():
    """Create a sample cotton image."""
    # Create directory if it doesn't exist
    os.makedirs("assets/samples", exist_ok=True)
    
    # Create a new image with light blue background (sky)
    img = Image.new('RGB', (400, 300), color=(220, 240, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw ground (field)
    draw.rectangle([(0, 200), (400, 300)], fill=(150, 180, 100))
    
    # Draw cotton plants
    for plant_idx in range(6):
        # Plant stem
        x_base = 50 + plant_idx * 60
        y_base = 220
        draw.line([(x_base, y_base), (x_base, y_base - 70)], fill=(100, 120, 80), width=3)
        
        # Branches
        for branch_idx in range(3):
            angle = -30 + branch_idx * 30
            length = 30 - abs(branch_idx - 1) * 5
            end_x = x_base + length * math.cos(math.radians(angle))
            end_y = y_base - 40 - length * math.sin(math.radians(angle))
            draw.line([(x_base, y_base - 40), (end_x, end_y)], fill=(100, 120, 80), width=2)
            
            # Cotton bolls
            draw_cotton_boll(draw, end_x, end_y)
    
    # Save the image
    output_path = "assets/samples/cotton_sample.jpg"
    img.save(output_path)
    print(f"Created cotton sample at {output_path}")
    return output_path

def draw_cotton_boll(draw, x, y):
    """Draw a cotton boll."""
    # Cotton is white and fluffy
    cotton_color = (250, 250, 250)
    
    # Draw the main boll
    for i in range(5):
        # Create a fluffy effect with multiple circles
        radius = 8 + random.randint(0, 3)
        offset_x = random.randint(-3, 3)
        offset_y = random.randint(-3, 3)
        draw.ellipse(
            [(x - radius + offset_x, y - radius + offset_y), 
             (x + radius + offset_x, y + radius + offset_y)], 
            fill=cotton_color
        )

if __name__ == "__main__":
    create_cotton_sample()