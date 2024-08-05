from tests import test_reductionist_json
from pathlib import Path 

mypath = Path(__file__).parent

test_reductionist_json.test_build_request(mypath)

