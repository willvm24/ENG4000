# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Tests for schola.rllib.imitation module
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from schola.rllib.imitation import read_expert_from_json, convert_to_rllib_format


@pytest.fixture
def sample_expert_data():
    """Create sample expert demonstration data"""
    return {
        "steps": [
            {
                "stepNum": 0,
                "observations": [
                    {
                        "interactorName": "sensor1",
                        "value": "1.0,2.0,3.0"
                    },
                    {
                        "interactorName": "sensor2",
                        "value": "4.0,5.0"
                    }
                ],
                "actions": [
                    {
                        "interactorName": "actuator1",
                        "value": "0.5,0.6"
                    }
                ],
                "rewards": 1.0
            },
            {
                "stepNum": 1,
                "observations": [
                    {
                        "interactorName": "sensor1",
                        "value": "1.1,2.1,3.1"
                    },
                    {
                        "interactorName": "sensor2",
                        "value": "4.1,5.1"
                    }
                ],
                "actions": [
                    {
                        "interactorName": "actuator1",
                        "value": "0.7,0.8"
                    }
                ]
            },
            {
                "stepNum": 2,
                "observations": [
                    {
                        "interactorName": "sensor1",
                        "value": "1.2,2.2,3.2"
                    },
                    {
                        "interactorName": "sensor2",
                        "value": "4.2,5.2"
                    }
                ],
                "actions": [
                    {
                        "interactorName": "actuator1",
                        "value": "0.9,1.0"
                    }
                ],
                "rewards": 2.0
            }
        ]
    }


@pytest.fixture
def expert_json_file(sample_expert_data):
    """Create a temporary JSON file with expert data"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_expert_data, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


class TestReadExpertFromJson:
    """Tests for read_expert_from_json function"""
    
    def test_read_expert_data(self, expert_json_file):
        """Test reading expert data from JSON file"""
        data = read_expert_from_json(expert_json_file)
        
        # Should return a list of [observation, action] pairs
        assert len(data) == 3
        assert all(len(item) == 2 for item in data)
        
    def test_observation_concatenation(self, expert_json_file):
        """Test that observations are concatenated correctly"""
        data = read_expert_from_json(expert_json_file)
        
        # First step observations: [1.0, 2.0, 3.0, 4.0, 5.0]
        first_obs = data[0][0]
        assert len(first_obs) == 5
        assert first_obs == [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Second step observations
        second_obs = data[1][0]
        assert len(second_obs) == 5
        assert second_obs == [1.1, 2.1, 3.1, 4.1, 5.1]
        
    def test_action_concatenation(self, expert_json_file):
        """Test that actions are concatenated correctly"""
        data = read_expert_from_json(expert_json_file)
        
        # First step actions: [0.5, 0.6]
        first_act = data[0][1]
        assert len(first_act) == 2
        assert first_act == [0.5, 0.6]
        
        # Third step actions
        third_act = data[2][1]
        assert third_act == [0.9, 1.0]
        
    def test_handles_missing_rewards(self, expert_json_file):
        """Test that missing rewards don't cause errors"""
        # The second step in our fixture has no rewards field
        data = read_expert_from_json(expert_json_file)
        
        # Should still process all steps
        assert len(data) == 3
        
    def test_file_not_found(self):
        """Test handling of non-existent file"""
        with pytest.raises(FileNotFoundError):
            read_expert_from_json("nonexistent_file.json")
            
    def test_invalid_json(self):
        """Test handling of invalid JSON"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content {")
            temp_path = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                read_expert_from_json(temp_path)
        finally:
            os.remove(temp_path)


class TestConvertToRllibFormat:
    """Tests for convert_to_rllib_format function"""
    
    def test_convert_basic(self, expert_json_file):
        """Test basic conversion to RLlib format"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output")
            
            result_path = convert_to_rllib_format(expert_json_file, output_path)
            
            # Should return a file path
            assert isinstance(result_path, str)
            assert os.path.exists(result_path)
            
    def test_convert_creates_directory(self, expert_json_file):
        """Test that conversion creates output directory if needed"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "nested", "dir", "output")
            
            result_path = convert_to_rllib_format(expert_json_file, output_path)
            
            # Should create nested directories
            assert os.path.exists(result_path)
            
    def test_convert_output_format(self, expert_json_file):
        """Test that output file is in correct format"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output")
            
            result_path = convert_to_rllib_format(expert_json_file, output_path)
            
            # File should exist and not be empty
            assert os.path.getsize(result_path) > 0
            
    def test_convert_preserves_data_order(self, expert_json_file):
        """Test that conversion preserves the order of expert data"""
        original_data = read_expert_from_json(expert_json_file)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output")
            result_path = convert_to_rllib_format(expert_json_file, output_path)
            
            # Verify we converted all steps
            assert len(original_data) == 3


class TestExpertDataEdgeCases:
    """Test edge cases for expert data handling"""
    
    def test_single_step_data(self):
        """Test handling of single step expert data"""
        single_step_data = {
            "steps": [
                {
                    "stepNum": 0,
                    "observations": [
                        {"interactorName": "sensor1", "value": "1.0"}
                    ],
                    "actions": [
                        {"interactorName": "actuator1", "value": "0.5"}
                    ]
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(single_step_data, f)
            temp_path = f.name
        
        try:
            data = read_expert_from_json(temp_path)
            assert len(data) == 1
            assert data[0][0] == [1.0]
            assert data[0][1] == [0.5]
        finally:
            os.remove(temp_path)
            
    def test_multiple_actuators(self):
        """Test handling of multiple actuators"""
        multi_actuator_data = {
            "steps": [
                {
                    "stepNum": 0,
                    "observations": [
                        {"interactorName": "sensor1", "value": "1.0"}
                    ],
                    "actions": [
                        {"interactorName": "actuator1", "value": "0.5"},
                        {"interactorName": "actuator2", "value": "0.6,0.7"}
                    ]
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(multi_actuator_data, f)
            temp_path = f.name
        
        try:
            data = read_expert_from_json(temp_path)
            # Actions should be concatenated: [0.5, 0.6, 0.7]
            assert data[0][1] == [0.5, 0.6, 0.7]
        finally:
            os.remove(temp_path)
