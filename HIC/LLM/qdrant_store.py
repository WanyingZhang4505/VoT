"""Persistent local Qdrant storage for completed HIC cases."""

import json
import math
import re
import uuid


MODEL_DIMENSIONS = {
    "BAAI/bge-m3": 1024,
}
MODEL_COLLECTION_STEMS = {
    "BAAI/bge-m3": "hic_bge_m3",
}
_COLLECTION_SUFFIX = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")


def collection_name_for(model_id, suffix=None):
    """Return a Qdrant collection name that cannot mix embedding models."""
    try:
        stem = MODEL_COLLECTION_STEMS[model_id]
    except KeyError:
        raise ValueError(f"Unsupported embedding model: {model_id}") from None
    if suffix is None:
        return stem
    if not isinstance(suffix, str) or not _COLLECTION_SUFFIX.fullmatch(suffix):
        raise ValueError("collection suffix must be lowercase letters, digits, '_' or '-'")
    return f"{stem}_{suffix}"


def _unit_vector(vector, dimension, label):
    if not isinstance(vector, (list, tuple)) or len(vector) != dimension:
        raise ValueError(f"{label} must contain exactly {dimension} dimensions")
    values = [float(value) for value in vector]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} must contain only finite values")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0 or not math.isfinite(norm):
        raise ValueError(f"{label} must have a non-zero finite norm")
    return [value / norm for value in values]


def _payloads_match(left, right):
    """Compare JSON payloads while tolerating Qdrant float round-trip noise."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _payloads_match(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _payloads_match(left_value, right_value)
            for left_value, right_value in zip(left, right)
        )
    if isinstance(left, float):
        return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
    return left == right


def _validate_targets(values, label, *, length=None):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{label} must contain at least one observed target value")
    if length is not None and len(values) != length:
        raise ValueError("prediction and observed targets must have the same number of target values")
    normalized = [float(value) for value in values]
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError(f"{label} must contain only finite numbers")
    return normalized


class QdrantHicStore:
    """One model-specific Qdrant collection containing completed HIC cases."""

    def __init__(
        self,
        *,
        model_id,
        collection_name=None,
        client=None,
        url="http://127.0.0.1:6333",
        models=None,
        scope_metadata=None,
        point_id_namespace=None,
    ):
        if model_id not in MODEL_DIMENSIONS:
            raise ValueError(f"Unsupported embedding model: {model_id}")
        if client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError:
                raise RuntimeError(
                    "Install qdrant-client==1.19.1 to use the local vector store."
                ) from None
            client = QdrantClient(url=url)
        if models is None:
            try:
                from qdrant_client.http import models as qdrant_models
            except ImportError:
                raise RuntimeError(
                    "Install qdrant-client==1.19.1 to use the local vector store."
                ) from None
            models = qdrant_models

        self.client = client
        self.models = models
        self.model_id = model_id
        self.dimension = MODEL_DIMENSIONS[model_id]
        self.collection_name = collection_name or collection_name_for(model_id)
        if scope_metadata is not None and (
            not isinstance(scope_metadata, dict) or not scope_metadata
        ):
            raise ValueError("scope_metadata must be a non-empty JSON object")
        if point_id_namespace is not None and (
            not isinstance(point_id_namespace, str) or not point_id_namespace.strip()
        ):
            raise ValueError("point_id_namespace must be non-empty text")
        self.scope_metadata = scope_metadata
        self.point_id_namespace = point_id_namespace

    def _scope_filter(self):
        if self.scope_metadata is None:
            return None
        conditions = [
            self.models.FieldCondition(
                key=f"metadata.{key}",
                match=self.models.MatchValue(value=value),
            )
            for key, value in sorted(self.scope_metadata.items())
        ]
        return self.models.Filter(must=conditions)

    def ensure_collection(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self.models.VectorParams(
                    size=self.dimension,
                    distance=self.models.Distance.COSINE,
                ),
            )
            return

        collection = self.client.get_collection(self.collection_name)
        vector_config = collection.config.params.vectors
        if isinstance(vector_config, dict):
            if len(vector_config) != 1:
                raise ValueError(
                    f"Collection {self.collection_name} must use one unnamed vector."
                )
            vector_config = next(iter(vector_config.values()))
        actual_dimension = getattr(vector_config, "size", None)
        if actual_dimension != self.dimension:
            raise ValueError(
                f"Collection {self.collection_name} has dimension {actual_dimension}; "
                f"{self.model_id} requires {self.dimension}."
            )

    @staticmethod
    def _point_id(collection_name, model_id, case_id, namespace=None):
        identity = f"qdrant-hic:{collection_name}:{model_id}:{case_id}"
        if namespace is not None:
            identity = f"{identity}:scope:{namespace}"
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                identity,
            )
        )

    def upsert_completed_case(
        self,
        *,
        case_id,
        summary,
        correction,
        embedding,
        observed_targets,
        prediction=None,
        cutoff_date=None,
        source=None,
        metadata=None,
        replace_seed_version=None,
    ):
        """Store one case only after its full outcome horizon and correction exist.

        Target and prediction values are validated but are not copied into the
        Qdrant payload; retrieval exposes the summary and correction only.
        """
        self.ensure_collection()
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("case_id must be a non-empty string")
        if replace_seed_version is not None and (
            not isinstance(replace_seed_version, str) or not replace_seed_version.strip()
        ):
            raise ValueError("replace_seed_version must be non-empty text when provided")
        if not isinstance(correction, str) or not correction.strip():
            raise ValueError("correction must be a non-empty string")
        try:
            json.dumps(summary, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            raise ValueError("summary must be finite JSON-compatible data") from None

        targets = _validate_targets(observed_targets, "observed_targets")
        if prediction is not None:
            _validate_targets(prediction, "prediction", length=len(targets))
        vector = _unit_vector(embedding, self.dimension, "embedding")
        payload = {
            "case_id": case_id,
            "summary": summary,
            "correction": correction.strip(),
            "embedding_model": self.model_id,
            "completed": True,
            "observed_target_count": len(targets),
        }
        if cutoff_date is not None:
            payload["cutoff_date"] = str(cutoff_date)
        if source is not None:
            payload["source"] = str(source)
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be a JSON object")
            try:
                payload["metadata"] = json.loads(
                    json.dumps(metadata, ensure_ascii=False, allow_nan=False)
                )
            except (TypeError, ValueError):
                raise ValueError("metadata must contain finite JSON-compatible data") from None
        if self.scope_metadata is not None:
            stored_metadata = payload.get("metadata", {})
            if any(
                stored_metadata.get(key) != value
                for key, value in self.scope_metadata.items()
            ):
                raise ValueError("metadata does not match the configured scope metadata")

        point_id = self._point_id(
            self.collection_name,
            self.model_id,
            case_id,
            namespace=self.point_id_namespace,
        )
        existing = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=True,
        )
        if existing:
            stored = existing[0]
            stored_vector = getattr(stored, "vector", None)
            if isinstance(stored_vector, dict):
                if len(stored_vector) != 1:
                    raise ValueError("Stored point unexpectedly contains named vectors")
                stored_vector = next(iter(stored_vector.values()))
            same_vector = (
                isinstance(stored_vector, (list, tuple))
                and len(stored_vector) == len(vector)
                and all(
                    math.isclose(float(left), right, rel_tol=1e-7, abs_tol=1e-6)
                    for left, right in zip(stored_vector, vector)
                )
            )
            if _payloads_match(getattr(stored, "payload", None), payload) and same_vector:
                return "existing"
            if replace_seed_version is not None:
                stored_payload = getattr(stored, "payload", None) or {}
                stored_metadata = stored_payload.get("metadata") or {}
                new_metadata = payload.get("metadata") or {}
                if (
                    stored_payload.get("case_id") == case_id
                    and stored_metadata.get("memory_version") == replace_seed_version
                    and stored_metadata.get("seeded_source_memory") is True
                    and new_metadata.get("memory_version") == replace_seed_version
                    and new_metadata.get("seeded_source_memory") is not True
                ):
                    point = self.models.PointStruct(
                        id=point_id, vector=vector, payload=payload
                    )
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=[point],
                        wait=True,
                    )
                    return "replaced_seed"
            raise ValueError(f"case_id {case_id!r} conflicts with stored case content")

        point = self.models.PointStruct(id=point_id, vector=vector, payload=payload)
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
            wait=True,
        )
        return "inserted"

    def upsert_validation_archive(
        self,
        *,
        case_id,
        summary,
        embedding,
        baseline_prediction,
        full_hic_prediction,
        observed_targets,
        metrics,
        retrieval_trace,
        metadata,
        source=None,
        details=None,
    ):
        """Archive a scored validation case in the shared collection.

        Validation archives are not correction memories. Callers must use a
        store scoped to ``split=validation`` and perform this write only after
        all paired predictions have been sealed and scored. Fit stores should
        be scoped to ``split=fit`` so these outcomes cannot enter retrieval.
        """
        self.ensure_collection()
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("case_id must be a non-empty string")
        if not isinstance(metadata, dict) or metadata.get("split") != "validation":
            raise ValueError("validation archive metadata must set split to 'validation'")
        targets = _validate_targets(observed_targets, "observed_targets")
        baseline = _validate_targets(
            baseline_prediction, "baseline_prediction", length=len(targets)
        )
        full_hic = _validate_targets(
            full_hic_prediction, "full_hic_prediction", length=len(targets)
        )
        vector = _unit_vector(embedding, self.dimension, "embedding")
        try:
            json.dumps(summary, ensure_ascii=False, allow_nan=False)
            json.dumps(metrics, ensure_ascii=False, allow_nan=False)
            json.dumps(retrieval_trace, ensure_ascii=False, allow_nan=False)
            json.dumps(details, ensure_ascii=False, allow_nan=False)
            safe_metadata = json.loads(
                json.dumps(metadata, ensure_ascii=False, allow_nan=False)
            )
        except (TypeError, ValueError):
            raise ValueError("validation archive fields must be finite JSON-compatible data") from None
        if self.scope_metadata is not None and any(
            safe_metadata.get(key) != value
            for key, value in self.scope_metadata.items()
        ):
            raise ValueError("metadata does not match the configured scope metadata")

        payload = {
            "case_id": case_id,
            "record_type": "validation_archive",
            "summary": summary,
            "embedding_model": self.model_id,
            "completed": True,
            "correction_status": "not_generated_for_validation",
            "observed_target_count": len(targets),
            "baseline_prediction": baseline,
            "full_hic_prediction": full_hic,
            "observed_targets": targets,
            "metrics": metrics,
            "retrieval_trace": retrieval_trace,
            "metadata": safe_metadata,
        }
        if source is not None:
            payload["source"] = str(source)
        if details is not None:
            payload["details"] = details

        point_id = self._point_id(
            self.collection_name,
            self.model_id,
            case_id,
            namespace=self.point_id_namespace,
        )
        existing = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=True,
        )
        if existing:
            stored = existing[0]
            stored_vector = getattr(stored, "vector", None)
            if isinstance(stored_vector, dict):
                if len(stored_vector) != 1:
                    raise ValueError("Stored point unexpectedly contains named vectors")
                stored_vector = next(iter(stored_vector.values()))
            same_vector = (
                isinstance(stored_vector, (list, tuple))
                and len(stored_vector) == len(vector)
                and all(
                    math.isclose(float(left), right, rel_tol=1e-7, abs_tol=2e-7)
                    for left, right in zip(stored_vector, vector)
                )
            )
            if getattr(stored, "payload", None) == payload and same_vector:
                return "existing"
            raise ValueError(f"case_id {case_id!r} conflicts with stored case content")

        point = self.models.PointStruct(id=point_id, vector=vector, payload=payload)
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
            wait=True,
        )
        return "inserted"

    def _query_filter(self, exclude_point_ids=None):
        scope_filter = self._scope_filter()
        excluded_ids = list(dict.fromkeys(str(point_id) for point_id in (exclude_point_ids or [])))
        if not excluded_ids:
            return scope_filter
        must = list(scope_filter.must or []) if scope_filter is not None else []
        must_not = list(scope_filter.must_not or []) if scope_filter is not None else []
        must_not.append(self.models.HasIdCondition(has_id=excluded_ids))
        return self.models.Filter(must=must or None, must_not=must_not)

    def search(self, query_vector, limit=1, *, exclude_point_ids=None):
        results = self.search_many(query_vector, limit=limit,
                                   exclude_point_ids=exclude_point_ids)
        return results[0] if results else None

    def search_many(self, query_vector, limit=1, *, exclude_point_ids=None):
        """Return ranked candidates with their applicability metadata."""
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        vector = _unit_vector(query_vector, self.dimension, "query_vector")
        query_kwargs = {
            "collection_name": self.collection_name,
            "query": vector,
            "limit": limit,
            "with_payload": True,
        }
        query_filter = self._query_filter(exclude_point_ids)
        if query_filter is not None:
            query_kwargs["query_filter"] = query_filter
        response = self.client.query_points(**query_kwargs)
        points = getattr(response, "points", [])
        results = []
        for point in points:
            payload = point.payload or {}
            result = {
                "point_id": str(point.id),
                "case_id": payload.get("case_id"),
                "summary": payload.get("summary"),
                "correction": payload.get("correction"),
                "score": float(point.score),
            }
            if "metadata" in payload:
                result["metadata"] = payload["metadata"]
            results.append(result)
        return results

    def count(self, *, exclude_point_ids=None):
        count_kwargs = {"collection_name": self.collection_name, "exact": True}
        query_filter = self._query_filter(exclude_point_ids)
        if query_filter is not None:
            count_kwargs["count_filter"] = query_filter
        return int(self.client.count(**count_kwargs).count)
