from typing import Any

from pydantic import BaseModel, Field, model_validator


class RagQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question or matching request.")
    context: str | None = Field(
        default=None,
        description="Optional backend-provided context used when no vector store is configured.",
    )
    obj_id: str | None = Field(default=None, description="Optional object id from perception/reconstruction.")
    room_context: dict[str, Any] | None = Field(
        default=None,
        description="Optional structured room/object metadata from the backend.",
    )
    model: str | None = Field(default=None, description="Optional OpenAI model override.")
    fit_candidates: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional dimension-filtered product candidates to ground the answer.",
    )


class RagQueryResponse(BaseModel):
    status: str
    answer: str
    model: str
    retrieval: dict[str, Any]


class Dimensions(BaseModel):
    x: float = Field(..., gt=0, description="Width on the x axis, in cm.")
    y: float = Field(..., gt=0, description="Depth on the y axis, in cm.")
    z: float = Field(..., gt=0, description="Height on the z axis, in cm.")


class ProductCandidate(BaseModel):
    id: str | int
    name: str | None = None
    product_name: str | None = None
    dimensions: Dimensions | None = None
    width_x: float | None = None
    depth_y: float | None = None
    height_z: float | None = None
    price: int | None = None
    category: str | None = None
    category_code: str | None = None
    product_url: str | None = None
    storage_path: str | None = Field(
        default=None,
        description="Firebase/GCS object path or full gs:// URI for the product asset.",
    )
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def normalize_schema_fields(self) -> "ProductCandidate":
        if self.name is None:
            self.name = self.product_name

        if self.dimensions is None and None not in (self.width_x, self.depth_y, self.height_z):
            self.dimensions = Dimensions(
                x=float(self.width_x),
                y=float(self.depth_y),
                z=float(self.height_z),
            )

        if self.category is None:
            self.category = self.category_code

        if self.name is None:
            raise ValueError("Either name or product_name is required.")
        if self.dimensions is None:
            raise ValueError("Either dimensions or width_x/depth_y/height_z is required.")
        return self


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class Position3D(BaseModel):
    x: float
    y: float
    z: float


class PerceptionObject(BaseModel):
    id: int
    label: str
    confidence: float | None = None
    bbox: BoundingBox | None = None
    position_3d: Position3D | None = None


class ParsedPerceptionObject(BaseModel):
    id: int
    category: str
    asset_filename: str | None = None
    estimated_width: float | None = None
    estimated_height: float | None = None
    confidence: float | None = None
    bbox: BoundingBox | None = None
    position_3d: Position3D | None = None


class FitQueryRequest(BaseModel):
    space: Dimensions
    products: list[ProductCandidate] = Field(..., min_length=1)
    clearance_cm: float = Field(
        default=0,
        ge=0,
        description="Required extra empty space subtracted from each available axis.",
    )
    allow_xy_rotation: bool = Field(
        default=True,
        description="Whether product width/depth can be swapped on the floor plane.",
    )


class FitProduct(BaseModel):
    id: str | int
    name: str
    dimensions: Dimensions
    fits: bool
    orientation: str | None
    remaining_cm: Dimensions | None
    storage_uri: str | None = None
    price: int | None = None
    category: str | None = None
    product_url: str | None = None
    relevance_score: float | None = None
    trace: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class FitQueryResponse(BaseModel):
    status: str
    space: Dimensions
    clearance_cm: float
    firebase_storage_base_uri: str
    fit_count: int
    products: list[FitProduct]


class FitRagQueryRequest(FitQueryRequest):
    query: str = Field(..., min_length=1, description="Question to answer using only fitting products.")
    context: str | None = None
    obj_id: str | None = None
    room_context: dict[str, Any] | None = None
    model: str | None = None


class PerceptionRagQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    perception_objects: list[PerceptionObject] = Field(..., min_length=1)
    products: list[ProductCandidate] | None = Field(
        default=None,
        description="Optional product rows. If omitted, matching-engine loads products from DB.",
    )
    target_object_id: int | None = None
    space: Dimensions | None = Field(
        default=None,
        description="Optional explicit available space. If omitted, the selected perception object's scale is used.",
    )
    clearance_cm: float = Field(default=0, ge=0)
    allow_xy_rotation: bool = True
    limit: int = Field(default=100, ge=1, le=200)
    room_context: dict[str, Any] | None = None
    model: str | None = None


class FitRagQueryResponse(BaseModel):
    status: str
    answer: str
    model: str
    retrieval: dict[str, Any]
    fit: FitQueryResponse
    perception: ParsedPerceptionObject | None = None
