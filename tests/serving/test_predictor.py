from xg_model.modelling.artifact import XgModel
from xg_model.serving.predictor import Situation, expected_goals

MODEL = XgModel.load()


def xg(**details):
    return expected_goals(MODEL, Situation(**details))


def test_result_is_a_probability():
    assert 0 < xg(x=108, y=40) < 1


def test_closer_shots_are_better():
    assert xg(x=110, y=40) > xg(x=90, y=40)


def test_central_shots_are_better_than_wide_ones():
    assert xg(x=108, y=40) > xg(x=108, y=20)


def test_headers_are_worse_than_shots_with_the_foot():
    assert xg(x=110, y=40, body_part="Head") < xg(x=110, y=40)


def test_defender_in_the_way_lowers_the_chance():
    assert xg(x=105, y=40, defenders=((110.0, 40.0),)) < xg(x=105, y=40)


def test_through_ball_raises_the_chance():
    ground_pass = xg(x=105, y=40, assist="Ground Pass")
    through_ball = xg(x=105, y=40, assist="Ground Pass", through_ball=True)

    assert through_ball > ground_pass
