import pytest
from fastapi import HTTPException

from app.common.jobs import validate_settings


def test_validate_settings_defaults_are_valid_photo():
    settings = validate_settings({}, 'guest')

    assert 'upload_type' not in settings
    assert settings['model_name'] == 'general'
    assert settings['target_resolution'] == '4k'
    assert settings['output_format'] == 'jpeg'


def test_validate_settings_rejects_unsupported_target_for_role():
    with pytest.raises(HTTPException, match='not allowed for this account'):
        validate_settings({'target_resolution': '8k'}, 'guest')


def test_validate_settings_accepts_high_target_for_admin():
    settings = validate_settings({'target_resolution': '8k'}, 'admin')

    assert settings['target_resolution'] == '8k'
