import pygame
import argparse
import sys
import os
import json
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Annotate an image.")
    parser.add_argument("image_path", help="The path to the image to display.")
    parser.add_argument("--scale", type=int, default=4, help="The scale factor for the image resolution.")
    parser.add_argument("--load", help="Path to a .sam_prompts.json file to load annotations from.")
    args = parser.parse_args()

    image_path = args.image_path
    resolution_scale = args.scale
    output_filename = os.path.splitext(image_path)[0] + ".sam_prompts.json"
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at '{image_path}'")
        sys.exit(1)

    pygame.init()
    pygame.font.init()

    try:
        image = pygame.image.load(image_path)
    except pygame.error as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    width, height = image.get_size()
    new_width, new_height = width // resolution_scale, height // resolution_scale
    image = pygame.transform.scale(image, (new_width, new_height))
    
    panel_width = 200
    screen = pygame.display.set_mode((new_width + panel_width, new_height))
    pygame.display.set_caption("Image Annotator")

    # Colors and font
    PANEL_COLOR = (50, 50, 50)
    TEXT_COLOR = (255, 255, 255)
    BUTTON_COLOR = (100, 100, 100)
    BUTTON_HOVER_COLOR = (150, 150, 150)
    font = pygame.font.SysFont(None, 50)
    small_font = pygame.font.SysFont(None, 36)

    object_id = 1
    points = []
    boxes = []
    annotation_history = []

    # Load existing annotations if a file is provided
    if args.load:
        if os.path.exists(args.load):
            points, boxes, annotation_history = load_annotations(args.load, resolution_scale)
            print(f"Loaded annotations from {args.load}")
        else:
            print(f"Warning: Annotation file not found at '{args.load}'. Starting fresh.")

    annotation_mode = "point"  # "point" or "box"
    drawing_box = False
    box_start_pos = None

    # Button setup
    button_y = 50
    minus_button_rect = pygame.Rect(new_width + 10, button_y, 40, 40)
    plus_button_rect = pygame.Rect(new_width + 150, button_y, 40, 40)

    mode_button_y = 320
    point_mode_button_rect = pygame.Rect(new_width + 10, mode_button_y, 85, 40)
    box_mode_button_rect = pygame.Rect(new_width + 105, mode_button_y, 85, 40)

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_s:
                    save_annotations(output_filename, points, boxes, resolution_scale)
                    print(f"Annotations saved to {output_filename}")
                elif event.key == pygame.K_c:
                    points.clear()
                    boxes.clear()
                    annotation_history.clear()
                    print("Cleared all annotations.")
                elif event.key == pygame.K_d:
                    if annotation_history:
                        last_annotation = annotation_history.pop()
                        if last_annotation["type"] == "point":
                            points.pop()
                        elif last_annotation["type"] == "box":
                            boxes.pop()
                        print("Removed last annotation.")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Handle clicks on buttons
                if event.button == 1:  # Left click
                    if minus_button_rect.collidepoint(mouse_pos):
                        object_id = max(1, object_id - 1)
                    elif plus_button_rect.collidepoint(mouse_pos):
                        object_id += 1
                    elif point_mode_button_rect.collidepoint(mouse_pos):
                        annotation_mode = "point"
                    elif box_mode_button_rect.collidepoint(mouse_pos):
                        annotation_mode = "box"

                # Handle clicks on the image area
                if mouse_pos[0] < new_width:
                    if annotation_mode == "point":
                        mods = pygame.key.get_mods()
                        is_negative = (event.button == 3) or \
                                      (event.button == 1 and (mods & pygame.KMOD_SHIFT or mods & pygame.KMOD_META))
                        label = 0 if is_negative else 1
                        if not (minus_button_rect.collidepoint(mouse_pos) or plus_button_rect.collidepoint(mouse_pos)):
                             point_data = {"pos": mouse_pos, "id": object_id, "label": label}
                             points.append(point_data)
                             annotation_history.append({"type": "point", "data": point_data})
                    elif annotation_mode == "box" and event.button == 1:
                        drawing_box = True
                        box_start_pos = mouse_pos
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if annotation_mode == "box" and drawing_box and event.button == 1:
                    drawing_box = False
                    if box_start_pos:
                        x1, y1 = box_start_pos
                        x2, y2 = mouse_pos
                        box_rect = pygame.Rect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
                        box_data = {"rect": box_rect, "id": object_id}
                        boxes.append(box_data)
                        annotation_history.append({"type": "box", "data": box_data})
                        box_start_pos = None


        # Drawing
        screen.fill((0, 0, 0)) # Black background
        screen.blit(image, (0, 0))

        # Draw markers for the current object ID
        for point in points:
            if point["id"] == object_id:
                color = (0, 255, 0) if point["label"] == 1 else (255, 0, 0) # Green for positive, Red for negative
                pygame.draw.circle(screen, color, point["pos"], 5)

        # Draw boxes for the current object ID
        for box in boxes:
            if box["id"] == object_id:
                pygame.draw.rect(screen, (0, 0, 255), box["rect"], 2)

        # Draw current box being drawn
        if drawing_box and box_start_pos:
            x1, y1 = box_start_pos
            x2, y2 = mouse_pos
            pygame.draw.rect(screen, (0, 0, 255), (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)), 2)

        # Draw the side panel
        pygame.draw.rect(screen, PANEL_COLOR, (new_width, 0, panel_width, new_height))
        
        # Display Object ID text
        id_label_text = small_font.render("Object ID:", True, TEXT_COLOR)
        screen.blit(id_label_text, (new_width + 10, 10))

        # Buttons
        # Minus button
        minus_color = BUTTON_HOVER_COLOR if minus_button_rect.collidepoint(mouse_pos) else BUTTON_COLOR
        pygame.draw.rect(screen, minus_color, minus_button_rect)
        minus_text = font.render("-", True, TEXT_COLOR)
        screen.blit(minus_text, (minus_button_rect.x + 12, minus_button_rect.y + 2))

        # Plus button
        plus_color = BUTTON_HOVER_COLOR if plus_button_rect.collidepoint(mouse_pos) else BUTTON_COLOR
        pygame.draw.rect(screen, plus_color, plus_button_rect)
        plus_text = font.render("+", True, TEXT_COLOR)
        screen.blit(plus_text, (plus_button_rect.x + 10, plus_button_rect.y + 2))

        # Display Object ID value
        id_value_text = font.render(str(object_id), True, TEXT_COLOR)
        id_value_rect = id_value_text.get_rect(center=((minus_button_rect.right + plus_button_rect.left) // 2, minus_button_rect.centery))
        screen.blit(id_value_text, id_value_rect)

        # Annotation mode buttons
        point_mode_color = BUTTON_HOVER_COLOR if annotation_mode == "point" else BUTTON_COLOR
        pygame.draw.rect(screen, point_mode_color, point_mode_button_rect)
        point_text = small_font.render("Point", True, TEXT_COLOR)
        screen.blit(point_text, (point_mode_button_rect.x + 15, point_mode_button_rect.y + 10))

        box_mode_color = BUTTON_HOVER_COLOR if annotation_mode == "box" else BUTTON_COLOR
        pygame.draw.rect(screen, box_mode_color, box_mode_button_rect)
        box_text = small_font.render("Box", True, TEXT_COLOR)
        screen.blit(box_text, (box_mode_button_rect.x + 25, box_mode_button_rect.y + 10))

        # Zoom window logic
        if mouse_pos[0] < new_width and mouse_pos[1] < new_height:
            zoom_level = 6
            zoom_area_size = 31  # Use an odd number for a clear center
            zoom_display_size = zoom_area_size * zoom_level # 31 * 6 = 186

            # Define the source rectangle from the image
            source_rect_x = max(0, mouse_pos[0] - zoom_area_size // 2)
            source_rect_y = max(0, mouse_pos[1] - zoom_area_size // 2)
            # Ensure the source rect does not go out of bounds
            if source_rect_x + zoom_area_size > new_width:
                source_rect_x = new_width - zoom_area_size
            if source_rect_y + zoom_area_size > new_height:
                source_rect_y = new_height - zoom_area_size
            
            source_rect = pygame.Rect(source_rect_x, source_rect_y, zoom_area_size, zoom_area_size)
            
            # Create a surface with the content to be zoomed
            zoom_surface = image.subsurface(source_rect)

            # Scale it up for the zoom effect
            zoomed_surface = pygame.transform.scale(zoom_surface, (zoom_display_size, zoom_display_size))

            # Draw existing points in the zoom window
            for point in points:
                if point["id"] == object_id and source_rect.collidepoint(point["pos"]):
                    # Calculate position relative to the zoom area
                    rel_x = point["pos"][0] - source_rect.x
                    rel_y = point["pos"][1] - source_rect.y
                    
                    # Scale position to the zoomed surface
                    zoom_x = rel_x * zoom_level
                    zoom_y = rel_y * zoom_level
                    
                    color = (0, 255, 0) if point["label"] == 1 else (255, 0, 0)
                    
                    point_rect = pygame.Rect(zoom_x, zoom_y, zoom_level, zoom_level)
                    pygame.draw.rect(zoomed_surface, color, point_rect)


            # Highlight the central pixel (cursor)
            center_pixel_size = zoom_level
            center_pixel_pos = (zoom_display_size // 2 - center_pixel_size // 2, zoom_display_size // 2 - center_pixel_size // 2)
            
            # Create a semi-transparent surface for the highlight
            highlight = pygame.Surface((center_pixel_size, center_pixel_size), pygame.SRCALPHA)
            highlight.fill((255, 0, 0, 100)) # Red, semi-transparent
            pygame.draw.rect(highlight, (255, 255, 255), highlight.get_rect(), 1) # White border
            
            zoomed_surface.blit(highlight, center_pixel_pos)

            # Blit the zoomed surface to the panel
            zoom_display_pos = (new_width + (panel_width - zoom_display_size) // 2, 120)
            screen.blit(zoomed_surface, zoom_display_pos)
            pygame.draw.rect(screen, TEXT_COLOR, (*zoom_display_pos, zoom_display_size, zoom_display_size), 1) # Border

        pygame.display.flip()

    pygame.quit()

def load_annotations(filename, scale):
    points = []
    boxes = []
    history = []
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        for key, value in data.items():
            obj_id = int(key.split('_')[-1])
            
            # Load points
            if "points" in value and "labels" in value:
                for i, p in enumerate(value["points"]):
                    pos = (p[0] // scale, p[1] // scale)
                    label = value["labels"][i]
                    point_data = {"pos": pos, "id": obj_id, "label": label}
                    points.append(point_data)
                    history.append({"type": "point", "data": point_data})

            # Load boxes
            if "boxes" in value:
                for b in value["boxes"]:
                    # Scale down the coordinates
                    rect = pygame.Rect(b[0] // scale, b[1] // scale, (b[2] - b[0]) // scale, (b[3] - b[1]) // scale)
                    box_data = {"rect": rect, "id": obj_id}
                    boxes.append(box_data)
                    history.append({"type": "box", "data": box_data})

    except (IOError, json.JSONDecodeError) as e:
        print(f"Error loading annotation file: {e}")

    return points, boxes, history

def save_annotations(filename, points, boxes, scale):
    annotations = {}
    
    all_object_ids = sorted(list(set([p["id"] for p in points] + [b["id"] for b in boxes])))

    for obj_id in all_object_ids:
        # Process points
        obj_points = [p["pos"] for p in points if p["id"] == obj_id]
        obj_labels = [p["label"] for p in points if p["id"] == obj_id]
        scaled_points = (np.array(obj_points) * scale).tolist() if obj_points else []

        # Process boxes
        obj_boxes = [b["rect"] for b in boxes if b["id"] == obj_id]
        scaled_boxes = []
        for box in obj_boxes:
            scaled_box = [box.left * scale, box.top * scale, box.right * scale, box.bottom * scale]
            scaled_boxes.append(scaled_box)

        annotations[f"object_{obj_id}"] = {
            "points": scaled_points,
            "labels": obj_labels,
            "boxes": scaled_boxes
        }

    with open(filename, "w") as f:
        json.dump(annotations, f, indent=4)

if __name__ == "__main__":
    main()

