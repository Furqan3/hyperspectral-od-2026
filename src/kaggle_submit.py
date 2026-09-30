"""Submit a CSV to the competition. Runs on /usr/bin/python3 (the interpreter that has kagglesdk).

  /usr/bin/python3 src/kaggle_submit.py submissions/sub.csv "description"
  /usr/bin/python3 src/kaggle_submit.py --list          # our submissions + scores
  /usr/bin/python3 src/kaggle_submit.py --leaderboard   # public standings

Token: read from ~/.kaggle/api_token (chmod 600) or $KAGGLE_API_TOKEN. Never commit it.
"""
import os
import sys

COMP = "hyperspectral-object-detection-challenge-2026"


def client():
    tok = os.environ.get("KAGGLE_API_TOKEN") or open(os.path.expanduser("~/.kaggle/api_token")).read().strip()
    os.environ["KAGGLE_API_TOKEN"] = tok
    from kagglesdk import KaggleClient
    return KaggleClient()


def submit(path, desc):
    import requests
    from kagglesdk.competitions.types.competition_api_service import (ApiCreateSubmissionRequest,
                                                                     ApiStartSubmissionUploadRequest)
    size = os.path.getsize(path)
    with client() as k:
        r = ApiStartSubmissionUploadRequest()
        r.competition_name, r.file_name = COMP, os.path.basename(path)
        r.content_length, r.last_modified_epoch_seconds = size, int(os.path.getmtime(path))
        up = k.competitions.competition_api_client.start_submission_upload(r)
        with open(path, "rb") as f:
            put = requests.put(up.create_url, data=f, timeout=600)
        put.raise_for_status()
        c = ApiCreateSubmissionRequest()
        c.competition_name, c.blob_file_tokens, c.submission_description = COMP, up.token, desc
        res = k.competitions.competition_api_client.create_submission(c)
        print("submitted:", getattr(res, "message", res))


def show(kind):
    from kagglesdk.competitions.types.competition_api_service import ApiListSubmissionsRequest
    with client() as k:
        if kind == "list":
            r = ApiListSubmissionsRequest(); r.competition_name = COMP
            for s in k.competitions.competition_api_client.list_submissions(r).submissions:
                print(f"{str(s.date)[:16]}  {s.public_score or '-':>9}  {s.file_name:34s} {s.description[:52]}")
        else:
            from kagglesdk.competitions.types.competition_api_service import ApiGetLeaderboardRequest as L
            r = L(); r.competition_name = COMP
            for i, s in enumerate(k.competitions.competition_api_client.get_leaderboard(r).submissions, 1):
                print(f"{i:3d}. {s.team_name[:30]:30s} {s.score}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] in ("--list", "--leaderboard"):
        show("list" if not a or a[0] == "--list" else "lb")
    else:
        submit(a[0], a[1] if len(a) > 1 else "auto submission")
