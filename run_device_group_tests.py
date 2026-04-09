#!/usr/bin/env python3
import sys
sys.path.insert(0, 'C:\\Users\\somca\\내문서\\project\\PIMS')

from src.utils.device_group_parser import DeviceGroupParser

results = []

try:
    # 테스트 1: 브래킷 접두사 파싱
    groups = DeviceGroupParser.parse(["[3_CO_PLC_B]QB 42"])
    assert "3_CO_PLC_B" in groups
    assert "[3_CO_PLC_B]QB 42" in groups["3_CO_PLC_B"]
    results.append("PASS: test_bracketed_column_uses_prefix")

    # 테스트 2: 기본 그룹
    groups = DeviceGroupParser.parse(["MB    118", "MD    356_1"])
    assert DeviceGroupParser.DEFAULT_GROUP in groups
    assert len(groups[DeviceGroupParser.DEFAULT_GROUP]) == 2
    results.append("PASS: test_unbracketed_column_goes_to_default")

    # 테스트 3: 혼합 컬럼
    cols = ["[3_CO_PLC_B]QB 42", "[3_ECS_PLC_B]MB    31", "MB    118"]
    groups = DeviceGroupParser.parse(cols)
    assert set(groups.keys()) == {"3_CO_PLC_B", "3_ECS_PLC_B", DeviceGroupParser.DEFAULT_GROUP}
    results.append("PASS: test_mixed_columns_split_correctly")

    # 테스트 4: 빈 컬럼
    assert DeviceGroupParser.parse([]) == {}
    results.append("PASS: test_empty_columns_returns_empty")

    # 테스트 5: group_id 메서드
    assert DeviceGroupParser.group_id("[3_CO_PLC_B]QB 42") == "3_CO_PLC_B"
    assert DeviceGroupParser.group_id("MB    118") == DeviceGroupParser.DEFAULT_GROUP
    results.append("PASS: test_group_id_single_column")

    # 테스트 6: 같은 그룹의 여러 신호
    cols = ["[3_CO_PLC_B]QB 42", "[3_CO_PLC_B]QB 43", "[3_CO_PLC_B]MB    10"]
    groups = DeviceGroupParser.parse(cols)
    assert len(groups["3_CO_PLC_B"]) == 3
    results.append("PASS: test_multiple_signals_same_group")

except Exception as e:
    results.append(f"FAIL: {e}")
    import traceback
    traceback.print_exc()

with open('C:\\Users\\somca\\내문서\\project\\PIMS\\test_results.txt', 'w', encoding='utf-8') as f:
    for result in results:
        f.write(result + '\n')
    f.write(f'\n총 {len(results)}개 테스트 완료\n')
    f.write('모든 테스트 성공' if len(results) == 6 else '일부 테스트 실패')

print('\n'.join(results))
