package com.cmb.tg.tgai.infrastructure.wecom.mapper;

import com.cmb.tg.tgai.infrastructure.wecom.po.ChatCustomerAuthPO;
import org.apache.ibatis.annotations.Param;

import java.util.List;

public interface ChatCustomerAuthMapper {

    int deleteByCustId(@Param("custId") String custId);

    int batchInsert(@Param("list") List<ChatCustomerAuthPO> list);
}
