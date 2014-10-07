# -*- coding: utf-8 -*-
"""Base class for testing the moe examples."""
import pytest


class MoeExampleTestCase(object):

    """Base class for testing the moe examples."""

    @pytest.fixture(autouse=True)
    def create_webapp(self):
        """Create a mocked webapp and store it in self.testapp."""
        from moe import main
        app = main({}, use_mongo='false')
        from webtest import TestApp
        self.testapp = TestApp(app)
