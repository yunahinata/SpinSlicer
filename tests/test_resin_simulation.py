"""Checks for the resin formulation screening model."""
from __future__ import annotations

import dataclasses

import pytest

from resin_simulation import (
    DEFAULT_MODEL_REGISTRY,
    Component,
    ModelPlugin,
    ModelRegistry,
    ProcessConditions,
    Recipe,
    calculate_recipe,
    composition_table,
    default_components,
    load_library,
    save_library,
)


def test_mass_and_volume_percentages_sum_to_one_hundred() -> None:
    components = default_components()
    recipe = Recipe(batch_mass_g=200, additives_g={"reactive_diluent": 30, "tpo_l": 3})
    rows = composition_table(recipe, components)

    assert sum(float(row["mass_g"]) for row in rows) == pytest.approx(200)
    assert sum(float(row["weight_pct"]) for row in rows) == pytest.approx(100)
    assert sum(float(row["volume_pct"]) for row in rows) == pytest.approx(100)
    assert next(row for row in rows if row["key"] == "standard_acrylate")["mass_g"] == pytest.approx(167)


def test_screening_outputs_explanations_and_sensitivity() -> None:
    components = default_components()
    recipe = Recipe()
    result = calculate_recipe(recipe, components)

    assert result.prediction("cure_depth").value is not None
    assert "Jacobs" in result.prediction("cure_depth").caveat
    assert result.prediction("viscosity").sources
    assert result.prediction("failure_risks").value
    assert any(item.name.startswith("+10%") for item in result.sensitivities)
    assert any(item.name == "Постотверждение" and item.printability_points is None for item in result.sensitivities)


def test_process_changes_exposure_window_and_layer_warning() -> None:
    components = default_components()
    recipe = Recipe(process=ProcessConditions(intensity_mw_cm2=6, exposure_s=12, layer_height_um=50))
    normal = calculate_recipe(recipe, components, sensitivity=False)
    recipe.process.layer_height_um = 100
    thicker = calculate_recipe(recipe, components, sensitivity=False)

    assert normal.prediction("cure_depth").value == thicker.prediction("cure_depth").value
    assert normal.prediction("exposure_window").value != thicker.prediction("exposure_window").value
    assert normal.prediction("layer_adhesion").value != thicker.prediction("layer_adhesion").value


def test_unknown_component_data_is_reported_instead_of_invented() -> None:
    components = {"custom_base": Component("custom_base", "Своя основа", "base", source="measured")}
    recipe = Recipe(base_id="custom_base", additives_g={})
    result = calculate_recipe(recipe, components, sensitivity=False)

    assert result.prediction("viscosity").value is None
    assert "вязкость" in result.prediction("viscosity").missing[0]
    assert result.prediction("cure_depth").value is None
    assert "Dp" in result.prediction("cure_depth").missing[0]
    assert result.prediction("shrinkage").value is None


def test_bad_dosage_and_component_values_are_rejected() -> None:
    components = default_components()
    with pytest.raises(ValueError, match="меньше массы партии"):
        Recipe(batch_mass_g=10, additives_g={"reactive_diluent": 10}).validate(components)
    with pytest.raises(ValueError, match="положительным"):
        Component("bad", "Плохой", "base", density_g_ml=-1).validate()


def test_library_round_trip_preserves_recipe_and_measurement_status(tmp_path) -> None:
    components = default_components()
    recipe = Recipe(name="Свой состав", process=ProcessConditions(measured=("wavelength_nm",)))
    path = tmp_path / "library.json"

    save_library(path, components, {recipe.name: recipe})
    loaded_components, loaded_recipes = load_library(path)

    assert loaded_components["standard_acrylate"].source == "estimate"
    assert loaded_recipes[recipe.name].process.measured == ("wavelength_nm",)
    assert loaded_recipes[recipe.name].additives_g == recipe.additives_g


def test_model_registry_accepts_and_dispatches_additional_implementations(monkeypatch) -> None:
    registry = ModelRegistry()
    plugin = ModelPlugin("custom", "Моя модель", "Пользовательская", calculate_recipe)
    registry.register(plugin)

    assert registry.get("custom") is plugin
    assert dataclasses.is_dataclass(plugin)
    dispatch_plugin = ModelPlugin(
        "test_dispatch",
        "Тестовая",
        "Проверка точки расширения",
        lambda recipe, components, *, sensitivity=True: calculate_recipe(
            dataclasses.replace(recipe, model_key="screening"), components, sensitivity=sensitivity
        ),
    )
    monkeypatch.setitem(DEFAULT_MODEL_REGISTRY._models, dispatch_plugin.key, dispatch_plugin)
    recipe = Recipe(model_key=dispatch_plugin.key)
    assert calculate_recipe(recipe, default_components(), sensitivity=False).prediction("viscosity").value
