package com.cmb.tg.tgai.infrastructure.wecom.mapper;

import com.cmb.tg.tgai.infrastructure.wecom.po.ChatEmployeePO;
import java.util.List;

public interface ChatEmployeeMapper {

    int batchUpsert(List<ChatEmployeePO> list);

    List<String> selectAllEmployeeIds();
}
