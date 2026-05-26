CREATE TABLE IF NOT EXISTS categories (
    id SERIAL PRIMARY KEY,
    category_name VARCHAR(100) NOT NULL,
    category_code VARCHAR(100) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS products (
    id BIGSERIAL PRIMARY KEY,
    category_id INTEGER NOT NULL REFERENCES categories(id),
    product_name VARCHAR(255) NOT NULL,
    price INTEGER NOT NULL,
    width_x REAL NOT NULL,
    depth_y REAL NOT NULL,
    height_z REAL NOT NULL,
    size_unit VARCHAR(20) NOT NULL DEFAULT 'cm',
    product_url VARCHAR(700) NOT NULL UNIQUE,
    source_site VARCHAR(100) NOT NULL DEFAULT 'IKEA',
    raw_size_text TEXT NULL,
    raw_json JSONB NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS product_images (
    id BIGSERIAL PRIMARY KEY,
    product_id BIGINT NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    image_url VARCHAR(700) NULL,
    image_filename VARCHAR(255) NOT NULL,
    image_mime VARCHAR(100) NOT NULL,
    image_data BYTEA NULL,
    image_description TEXT NULL,
    is_main BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO categories (category_name, category_code)
VALUES
    ('책상', 'desk'),
    ('의자', 'chair'),
    ('선반', 'shelf')
ON CONFLICT (category_code) DO UPDATE
SET category_name = EXCLUDED.category_name,
    updated_at = CURRENT_TIMESTAMP;
