package com.cmb.tg.tgai.infrastructure.wecom.mapper;

import com.cmb.tg.tgai.infrastructure.wecom.po.ChatCustomerPO;
import java.util.List;

public interface ChatCustomerMapper {

    int batchUpsert(List<ChatCustomerPO> list);

    List<String> selectAllCustIds();
}
