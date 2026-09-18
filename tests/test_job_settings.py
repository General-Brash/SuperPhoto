import pytest
from fastapi import HTTPException

from app.common.jobs import validate_settings


def test_validate_settings_defaults_to_photo():
    settings = validate_settings({}, 'guest')

    assert settings['upload_type'] == 'photo'


def test_validate_settings_rejects_unknown_upload_type():
    with pytest.raises(HTTPException, match='Unsupported upload type'):
        validate_settings({'upload_type': 'document'}, 'guest')


def test_validate_settings_allows_video_for_admin():
    settings = validate_settings(
        {'upload_type': 'video', 'target_resolution': '1k', 'output_format': 'mp4'}, 'admin'
    )

    assert settings['upload_type'] == 'video'
    assert settings['target_resolution'] == '1k'


def test_validate_settings_restricts_video_to_admin():
    with pytest.raises(HTTPException, match='restricted to administrators'):
        validate_settings({'upload_type': 'video', 'target_resolution': '1k', 'output_format': 'mp4'}, 'user')


@pytest.mark.parametrize('output_format', ['mp4', 'webm'])
def test_validate_settings_accepts_video_output_formats(output_format):
    settings = validate_settings(
        {'upload_type': 'video', 'target_resolution': '4k', 'output_format': output_format}, 'admin'
    )

    assert settings['output_format'] == output_format


def test_validate_settings_rejects_image_format_for_video():
    with pytest.raises(HTTPException, match='MP4 or WebM'):
        validate_settings({'upload_type': 'video', 'target_resolution': '2k', 'output_format': 'png'}, 'admin')
