"""The action table is a checkpoint contract: order and casing must not drift."""

from __future__ import annotations

import unittest

from smb1_ppo import actions


class ActionTableTests(unittest.TestCase):
    def test_table_matches_the_documented_mapping(self) -> None:
        self.assertEqual(
            actions.ACTION_NAMES,
            ("NOOP", "right", "right+A", "right+B", "right+A+B", "left", "left+A"),
        )
        self.assertEqual(actions.ACTION_COUNT, 7)
        self.assertLessEqual(actions.ACTION_COUNT, actions.MAX_ACTIONS)

    def test_each_action_presses_only_its_named_buttons(self) -> None:
        for name, buttons in actions.ACTION_SPECS:
            expected = [] if name == "NOOP" else name.split("+")
            self.assertEqual(list(buttons), expected, name)

    def test_no_action_mixes_left_and_right(self) -> None:
        for name, buttons in actions.ACTION_SPECS:
            self.assertFalse({"left", "right"} <= set(buttons), name)

    def test_the_table_is_independent_of_the_rainbow_action_space(self) -> None:
        # The Ape-X Rainbow system encodes 7 bases x 3 durations = 21 macros.
        self.assertEqual(actions.ACTION_COUNT, 7)
        self.assertNotEqual(actions.ACTION_COUNT, 21)

    def test_button_names_are_accepted_by_nes_py(self) -> None:
        from nes_py.wrappers import JoypadSpace

        for name, buttons in actions.ACTION_SPECS:
            for button in buttons:
                self.assertIn(button, JoypadSpace._button_map, f"{name}: {button}")

    def test_validate_action_rejects_out_of_range_and_non_integers(self) -> None:
        self.assertEqual(actions.validate_action(0), 0)
        self.assertEqual(actions.validate_action(6), 6)
        for bad in (-1, 7, 100):
            with self.assertRaises(ValueError):
                actions.validate_action(bad)
        for bad in (True, "right", 1.0, None):
            with self.assertRaises(TypeError):
                actions.validate_action(bad)

    def test_helpers_agree_with_the_table(self) -> None:
        self.assertEqual(actions.action_name(2), "right+A")
        self.assertEqual(actions.buttons_for(4), ["right", "A", "B"])
        actions.buttons_for(4).append("left")
        self.assertEqual(actions.buttons_for(4), ["right", "A", "B"])
        self.assertEqual(actions.action_counts([0, 0, 1, 6, 6, 6]), [2, 1, 0, 0, 0, 0, 3])


if __name__ == "__main__":
    unittest.main()
