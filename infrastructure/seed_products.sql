WITH desk_category AS (
    SELECT id FROM categories WHERE category_code = 'desk'
),
chair_category AS (
    SELECT id FROM categories WHERE category_code = 'chair'
),
shelf_category AS (
    SELECT id FROM categories WHERE category_code = 'shelf'
),
upsert_products AS (
    INSERT INTO products (
        category_id,
        product_name,
        price,
        width_x,
        depth_y,
        height_z,
        product_url,
        source_site,
        raw_size_text
    )
    VALUES
        ((SELECT id FROM desk_category), 'MICKE desk', 89900, 105, 50, 72, 'https://example.com/products/micke-desk', 'TEST', '105x50x72 cm'),
        ((SELECT id FROM desk_category), 'Large studio desk', 159000, 140, 70, 75, 'https://example.com/products/large-studio-desk', 'TEST', '140x70x75 cm'),
        ((SELECT id FROM desk_category), 'Compact laptop table', 49000, 80, 45, 70, 'https://example.com/products/compact-laptop-table', 'TEST', '80x45x70 cm'),
        ((SELECT id FROM chair_category), 'Simple dining chair', 39000, 45, 50, 82, 'https://example.com/products/simple-dining-chair', 'TEST', '45x50x82 cm'),
        ((SELECT id FROM shelf_category), 'Narrow shelf unit', 69000, 60, 30, 150, 'https://example.com/products/narrow-shelf-unit', 'TEST', '60x30x150 cm')
    ON CONFLICT (product_url) DO UPDATE
    SET
        product_name = EXCLUDED.product_name,
        price = EXCLUDED.price,
        width_x = EXCLUDED.width_x,
        depth_y = EXCLUDED.depth_y,
        height_z = EXCLUDED.height_z,
        raw_size_text = EXCLUDED.raw_size_text,
        updated_at = CURRENT_TIMESTAMP
    RETURNING id, product_url
)
INSERT INTO product_images (
    product_id,
    image_filename,
    image_mime,
    image_description,
    is_main
)
SELECT
    p.id,
    CASE p.product_url
        WHEN 'https://example.com/products/micke-desk' THEN 'products/desk/micke.jpg'
        WHEN 'https://example.com/products/large-studio-desk' THEN 'products/desk/large_studio.jpg'
        WHEN 'https://example.com/products/compact-laptop-table' THEN 'products/desk/compact_laptop.jpg'
        WHEN 'https://example.com/products/simple-dining-chair' THEN 'products/chair/simple_dining_chair.jpg'
        WHEN 'https://example.com/products/narrow-shelf-unit' THEN 'products/shelf/narrow_shelf.jpg'
    END,
    'image/jpeg',
    CASE p.product_url
        WHEN 'https://example.com/products/micke-desk' THEN 'Compact white desk suitable for small rooms.'
        WHEN 'https://example.com/products/large-studio-desk' THEN 'Large desk for a wider studio layout.'
        WHEN 'https://example.com/products/compact-laptop-table' THEN 'Small laptop table for tight spaces.'
        WHEN 'https://example.com/products/simple-dining-chair' THEN 'Simple chair with neutral styling.'
        WHEN 'https://example.com/products/narrow-shelf-unit' THEN 'Tall narrow shelf for vertical storage.'
    END,
    TRUE
FROM upsert_products p
WHERE NOT EXISTS (
    SELECT 1
    FROM product_images img
    WHERE img.product_id = p.id
      AND img.is_main = TRUE
);
