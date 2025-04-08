"""
Create a sample wheat image for testing.
"""

from PIL import Image, ImageDraw
import os

def create_wheat_sample():
    """Create a sample wheat image."""
    # Create directory if it doesn't exist
    os.makedirs("assets/samples", exist_ok=True)
    
    # Create a new image with white background
    img = Image.new('RGB', (400, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw wheat grains
    wheat_color = (220, 180, 100)
    
    # Draw multiple grains
    for i in range(40):
        x = 50 + (i % 8) * 40
        y = 50 + (i // 8) * 40
        
        # Draw a grain (oval)
        draw.ellipse((x, y, x + 30, y + 15), fill=wheat_color, outline=(200, 160, 80))
    
    # Add some shading
    for i in range(10):
        x = 60 + (i % 5) * 60
        y = 70 + (i // 5) * 60
        
        # Draw a slightly darker grain
        darker = (190, 150, 70)
        draw.ellipse((x, y, x + 30, y + 15), fill=darker, outline=(170, 130, 50))
    
    # Save the image
    output_path = "assets/samples/wheat_sample.jpg"
    img.save(output_path)
    print(f"Created wheat sample at {output_path}")
    return output_path

if __name__ == "__main__":
    create_wheat_sample()