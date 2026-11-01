import unittest

import numpy as np

from evaluation.vcoco_sets import assemble, average_precision, evaluate, overlaps, score_image


class SetTests(unittest.TestCase):
    actions = ["cut", "hold", "point"]
    roles = [["agent", "obj", "instr"], ["agent", "obj"], ["agent", "obj"]]
    person = [10, 10, 100, 100]
    obj = [120, 10, 150, 50]
    instr = [160, 10, 180, 40]

    def row(self, person=None, **slots):
        return {"person_box": self.person if person is None else person, **slots}

    def gt(self, missing=False):
        return {"boxes": np.array([self.person, self.obj, self.instr]),
                "gt_classes": np.array([1, 2, 3]),
                "gt_actions": np.array([[1, 1, 0], [-1, -1, -1], [-1, -1, -1]]),
                "gt_role_id": np.array([[[1, -1 if missing else 2], [1, -1], [-1, -1]],
                                        [[-1, -1]] * 3, [[-1, -1]] * 3])}

    def event(self, entities=None, score=.8, person=None):
        return {"person": tuple(self.person if person is None else person), "action": 0,
                "entities": [tuple(self.obj), tuple(self.instr)] if entities is None else entities, "score": score}

    def test_fragmented_person_merged(self):
        rows = [self.row(cut_obj=self.obj + [.8]),
                self.row([11, 10, 101, 100], cut_instr=self.instr + [.7])]
        events, audit = assemble(rows, self.actions, self.roles)
        self.assertEqual(audit["person_clusters"], 1)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0]["score"], .7)
        self.assertEqual(score_image(self.gt(), events, self.roles)[0][0][2:], (True, True))

    def test_input_order_and_duplicate_invariant(self):
        rows = [self.row(cut_obj=self.obj + [.8]), self.row([11, 10, 101, 100], cut_instr=self.instr + [.7])]
        a = assemble(rows, self.actions, self.roles)[0]
        self.assertEqual(a, assemble(rows[::-1] + rows, self.actions, self.roles)[0])

    def test_far_people_not_merged(self):
        rows = [self.row(cut_obj=self.obj + [.8]), self.row([200, 200, 290, 290], cut_instr=self.instr + [.7])]
        events, audit = assemble(rows, self.actions, self.roles)
        self.assertEqual(audit["person_clusters"], 2)
        self.assertFalse(events)

    def test_nontransitive_representatives(self):
        boxes = [[10, 10, 109, 109], [25, 10, 124, 109], [40, 10, 139, 109]]
        self.assertGreater(overlaps(boxes[:1], boxes[1:2])[0, 0], .7)
        rows = [self.row(b, hold_obj=self.obj + [s]) for b, s in zip(boxes, [.9, .8, .7])]
        self.assertEqual(assemble(rows, self.actions, self.roles)[1]["person_clusters"], 2)

    def test_best_per_slot_and_min_score(self):
        rows = [self.row(cut_obj=self.obj + [.9], cut_instr=self.instr + [.4]),
                self.row(cut_obj=self.instr + [.2], cut_instr=self.instr + [.8])]
        events, _ = assemble(rows, self.actions, self.roles)
        self.assertAlmostEqual(events[0]["score"], .8)
        self.assertEqual(events[0]["entities"][0], tuple(self.obj))

    def test_wrong_one_slot_fails_whole_set(self):
        event = self.event([tuple(self.obj), tuple(self.obj)])
        self.assertEqual(score_image(self.gt(), [event], self.roles)[0][0][2:], (False, False))

    def test_s1_s2_missing_filler(self):
        row = score_image(self.gt(True), [self.event()], self.roles)[0][0]
        self.assertEqual(row[2:], (False, True))
        event = self.event([tuple(self.obj), None])
        self.assertEqual(score_image(self.gt(True), [event], self.roles)[0][0][2:], (True, True))
        self.assertEqual(score_image(self.gt(), [event], self.roles)[0][0][2:], (False, False))

    def test_null_formats_and_missing_not_oracle_completed(self):
        for null in ([0, 0, 0, 0], [float("nan")] * 4):
            events, _ = assemble([self.row(cut_obj=self.obj + [.8], cut_instr=null + [.7])], self.actions, self.roles)
            self.assertIsNone(events[0]["entities"][1])
        self.assertFalse(assemble([self.row(cut_obj=self.obj + [.8])], self.actions, self.roles)[0])

    def test_one_use_duplicate_matching(self):
        rows, _ = score_image(self.gt(), [self.event(), self.event(score=.7)], self.roles)
        self.assertEqual([r[2:] for r in rows], [(True, True), (False, False)])

    def test_ignored_person_and_no_person(self):
        gt = self.gt()
        gt["gt_actions"][0] = -1
        self.assertEqual(score_image(gt, [self.event()], self.roles), ([], 1))
        gt["gt_classes"][0] = 2
        self.assertEqual(score_image(gt, [self.event()], self.roles)[0][0][2:], (False, False))

    def test_point_and_agent_only_excluded(self):
        rows = [self.row(point_obj=self.obj + [.9], hold_agent=.9)]
        self.assertFalse(assemble(rows, self.actions, self.roles)[0])

    def test_ap_ties_no_predictions_and_missing_recall(self):
        self.assertEqual(average_precision([], 2), 0.)
        self.assertEqual(average_precision([(.9, True)], 2), .5)
        self.assertEqual(average_precision([(.9, True), (.9, False)], 1), .5)
        self.assertEqual(average_precision([(.9, False), (.8, True)], 1), .5)

    def test_invalid_nonfinite_fields(self):
        with self.assertRaises(ValueError):
            assemble([self.row(cut_obj=self.obj + [float("inf")])], self.actions, self.roles)
        with self.assertRaises(ValueError):
            assemble([self.row(cut_obj=[float("nan"), 1, 2, 3, .8])], self.actions, self.roles)

    def test_end_to_end_action_mean_not_slot_mean(self):
        gt = dict(self.gt(), id=1)
        rows = [self.row(cut_obj=self.obj + [.8], cut_instr=self.obj + [.7], hold_obj=self.obj + [.9])]
        result = evaluate([gt], {1: rows}, self.actions, self.roles)
        self.assertEqual((result["images"], result["actions"], result["gt_sets"]), (1, 2, 2))
        for score in result["results"].values():
            self.assertEqual(score["set_map_percent"], 50.)
            self.assertEqual(score["dual_slot_map_percent"], 0.)
            self.assertEqual(score["single_slot_map_percent"], 100.)

    def test_images_without_predictions_retain_gt_support(self):
        database = [dict(self.gt(), id=i) for i in (1, 2)]
        rows = [self.row(cut_obj=self.obj + [.8], cut_instr=self.instr + [.7], hold_obj=self.obj + [.9])]
        result = evaluate(database, {1: rows}, self.actions, self.roles)
        self.assertEqual(result["gt_sets"], 4)
        self.assertEqual(result["results"]["scenario_1"]["set_map_percent"], 50.)

    def test_unknown_image_id_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([dict(self.gt(), id=1)], {2: []}, self.actions, self.roles)


if __name__ == "__main__":
    unittest.main()
