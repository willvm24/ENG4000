# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Tests for schola.core.protocols.base module
"""

import pytest
from schola.core.protocols.base import (
    AutoResetType,
    BaseProtocol,
    BaseProtocolMixin,
    BaseRLProtocol,
    BaseImitationProtocol,
)

class TestBaseProtocol:
    """Tests for BaseProtocol"""
    
    def test_properties_default(self):
        """Test that properties returns empty dict by default"""
        protocol = BaseProtocol()
        
        assert protocol.properties == {}
        assert isinstance(protocol.properties, dict)
        
    def test_properties_returns_new_dict(self):
        """Test that properties returns a new dict each time"""
        protocol = BaseProtocol()
        
        props1 = protocol.properties
        props2 = protocol.properties
        
        # Should return equal dicts
        assert props1 == props2
        # But they can be different objects
        assert isinstance(props1, dict)
        assert isinstance(props2, dict)


class ConcreteProtocol(BaseProtocol):
    """A concrete implementation of BaseProtocol for testing"""
    
    def __init__(self):
        self._active = False
        self._definition = None
        
    def close(self):
        self._active = False
        
    def start(self):
        self._active = True
        
    def __bool__(self):
        return self._active
        
    def send_startup_msg(self, *args, **kwargs):
        pass
        
    def get_definition(self, *args, **kwargs):
        return self._definition


class TestBaseProtocolImplementation:
    """Tests for BaseProtocol implementation"""
    
    def test_close_method_exists(self):
        """Test that close method can be called"""
        protocol = ConcreteProtocol()
        protocol.start()
        assert protocol._active
        
        protocol.close()
        assert not protocol._active
        
    def test_start_method_exists(self):
        """Test that start method can be called"""
        protocol = ConcreteProtocol()
        assert not protocol._active
        
        protocol.start()
        assert protocol._active
        
    def test_bool_method(self):
        """Test __bool__ method"""
        protocol = ConcreteProtocol()
        
        assert not protocol
        
        protocol.start()
        assert protocol
        
        protocol.close()
        assert not protocol
        
    def test_send_startup_msg(self):
        """Test send_startup_msg method"""
        protocol = ConcreteProtocol()
        # Should not raise
        protocol.send_startup_msg()
        protocol.send_startup_msg("arg")
        protocol.send_startup_msg(kwarg="value")
        
    def test_get_definition(self):
        """Test get_definition method"""
        protocol = ConcreteProtocol()
        result = protocol.get_definition()
        assert result is None


class TestBaseProtocolMixin:
    """Tests for BaseProtocolMixin"""
    
    def test_on_close_default(self):
        """Test that on_close does nothing by default"""
        mixin = BaseProtocolMixin()
        # Should not raise
        mixin.on_close()
        
    def test_on_start_default(self):
        """Test that on_start does nothing by default"""
        mixin = BaseProtocolMixin()
        # Should not raise
        mixin.on_start()
        
    def test_mixin_properties_default(self):
        """Test that mixin_properties returns empty dict by default"""
        mixin = BaseProtocolMixin()
        
        assert mixin.mixin_properties == {}
        assert isinstance(mixin.mixin_properties, dict)


class ConcreteMixin(BaseProtocolMixin):
    """A concrete implementation of BaseProtocolMixin for testing"""
    
    def __init__(self):
        self.closed = False
        self.started = False
        
    def on_close(self):
        self.closed = True
        
    def on_start(self):
        self.started = True
        
    @property
    def mixin_properties(self):
        return {"test_prop": "test_value"}


class TestBaseProtocolMixinImplementation:
    """Tests for BaseProtocolMixin implementation"""
    
    def test_on_close_can_be_overridden(self):
        """Test that on_close can be overridden"""
        mixin = ConcreteMixin()
        assert not mixin.closed
        
        mixin.on_close()
        assert mixin.closed
        
    def test_on_start_can_be_overridden(self):
        """Test that on_start can be overridden"""
        mixin = ConcreteMixin()
        assert not mixin.started
        
        mixin.on_start()
        assert mixin.started
        
    def test_mixin_properties_can_be_overridden(self):
        """Test that mixin_properties can be overridden"""
        mixin = ConcreteMixin()
        
        props = mixin.mixin_properties
        assert props == {"test_prop": "test_value"}


class ConcreteRLProtocol(BaseRLProtocol):
    """A concrete implementation of BaseRLProtocol for testing"""
    
    def __init__(self):
        self._active = False
        self.auto_reset = None
        
    def close(self):
        self._active = False
        
    def start(self):
        self._active = True
        
    def __bool__(self):
        return self._active
        
    def send_startup_msg(self, auto_reset_type=AutoResetType.SAME_STEP):
        self.auto_reset = auto_reset_type
        
    def get_definition(self):
        return ([], [], {}, {})
        
    def send_reset_msg(self, seeds=None, options=None):
        return ([], [])
        
    def send_action_msg(self, actions, action_space):
        return ([], [], [], [], [], {}, {})


class TestBaseRLProtocol:
    """Tests for BaseRLProtocol"""
    
    def test_send_startup_msg_with_auto_reset(self):
        """Test send_startup_msg accepts auto_reset_type"""
        protocol = ConcreteRLProtocol()
        
        protocol.send_startup_msg(AutoResetType.DISABLED)
        assert protocol.auto_reset == AutoResetType.DISABLED
        
        protocol.send_startup_msg(AutoResetType.NEXT_STEP)
        assert protocol.auto_reset == AutoResetType.NEXT_STEP
        
    def test_send_startup_msg_default_auto_reset(self):
        """Test send_startup_msg has default auto_reset_type"""
        protocol = ConcreteRLProtocol()
        
        protocol.send_startup_msg()
        assert protocol.auto_reset == AutoResetType.SAME_STEP
        
    def test_get_definition_returns_tuple(self):
        """Test get_definition returns tuple with 4 elements"""
        protocol = ConcreteRLProtocol()
        
        result = protocol.get_definition()
        assert isinstance(result, tuple)
        assert len(result) == 4
        
    def test_send_reset_msg(self):
        """Test send_reset_msg accepts seeds and options"""
        protocol = ConcreteRLProtocol()
        
        result = protocol.send_reset_msg(seeds=[1, 2, 3], options=[{}, {}])
        assert isinstance(result, tuple)
        assert len(result) == 2
        
    def test_send_reset_msg_no_args(self):
        """Test send_reset_msg works with no arguments"""
        protocol = ConcreteRLProtocol()
        
        result = protocol.send_reset_msg()
        assert isinstance(result, tuple)
        
    def test_send_action_msg(self):
        """Test send_action_msg returns tuple with 7 elements"""
        protocol = ConcreteRLProtocol()
        
        result = protocol.send_action_msg({}, {})
        assert isinstance(result, tuple)
        assert len(result) == 7


class ConcreteImitationProtocol(BaseImitationProtocol):
    """A concrete implementation of BaseImitationProtocol for testing"""
    
    def __init__(self):
        self._active = False
        
    def close(self):
        self._active = False
        
    def start(self):
        self._active = True
        
    def __bool__(self):
        return self._active
        
    def send_startup_msg(self, seeds=None, options=None):
        pass
        
    def get_definition(self):
        return ([], {}, {}, {})
        
    def get_data(self):
        return ([], [], [], [], [], {}, {}, {})


class TestBaseImitationProtocol:
    """Tests for BaseImitationProtocol"""
    
    def test_send_startup_msg_with_seeds_and_options(self):
        """Test send_startup_msg accepts seeds and options"""
        protocol = ConcreteImitationProtocol()
        
        # Should not raise
        protocol.send_startup_msg(seeds=[1, 2, 3], options=[{}, {}])
        
    def test_send_startup_msg_no_args(self):
        """Test send_startup_msg works with no arguments"""
        protocol = ConcreteImitationProtocol()
        
        # Should not raise
        protocol.send_startup_msg()
        
    def test_get_definition_returns_tuple(self):
        """Test get_definition returns tuple with 4 elements"""
        protocol = ConcreteImitationProtocol()
        
        result = protocol.get_definition()
        assert isinstance(result, tuple)
        assert len(result) == 4
        
    def test_get_data_returns_tuple(self):
        """Test get_data returns tuple with 8 elements"""
        protocol = ConcreteImitationProtocol()
        
        result = protocol.get_data()
        assert isinstance(result, tuple)
        assert len(result) == 8
        
    def test_inheritance(self):
        """Test that BaseImitationProtocol inherits from BaseProtocol"""
        assert issubclass(BaseImitationProtocol, BaseProtocol)


class TestProtocolInheritance:
    """Tests for protocol inheritance relationships"""
    
    def test_rl_protocol_inherits_base_protocol(self):
        """Test BaseRLProtocol inherits from BaseProtocol"""
        assert issubclass(BaseRLProtocol, BaseProtocol)
        
    def test_imitation_protocol_inherits_base_protocol(self):
        """Test BaseImitationProtocol inherits from BaseProtocol"""
        assert issubclass(BaseImitationProtocol, BaseProtocol)
