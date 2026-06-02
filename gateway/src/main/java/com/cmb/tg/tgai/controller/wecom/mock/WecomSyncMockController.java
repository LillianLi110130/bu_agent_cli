package com.cmb.tg.tgai.controller.wecom.mock;

import com.cmb.tg.tgai.infrastructure.common.rest.wecom.dto.response.CustomerBaseInfoResponseDTO;
import com.cmb.tg.tgai.infrastructure.common.rest.wecom.dto.response.EmployeeInfoResponseDTO;
import com.cmb.tg.tgai.infrastructure.common.rest.wecom.dto.response.GroupBaseInfoResponseDTO;
import org.springframework.util.StringUtils;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/mock/wecom")
public class WecomSyncMockController {

    private static final String SUCCESS_RTN_COD = "0000200";
    private static final String SUCCESS_ERR_MSG = "success";
    private static final String EMPLOYEE_ID = "80274658";
    private static final String CUSTOMER_ID = "wmMN17BwAAAAGwBk1Q6E1BBWNz4jMI6A";

    @PostMapping("/group/group-base-info")
    public GroupBaseInfoResponseDTO queryGroupBaseInfo(@RequestBody final Map<String, Object> request) {
        String groupId = valueOf(request.get("groupId"));

        GroupBaseInfoResponseDTO responseDTO = buildGroupSuccess();
        if (!StringUtils.hasText(groupId) || groupId.contains("missing")) {
            GroupBaseInfoResponseDTO.GroupInfoDTO emptyGroup = new GroupBaseInfoResponseDTO.GroupInfoDTO();
            emptyGroup.setGroupMemberList(Collections.emptyList());
            responseDTO.setInfBdy(Collections.singletonList(emptyGroup));
            return responseDTO;
        }

        GroupBaseInfoResponseDTO.GroupInfoDTO groupInfoDTO = new GroupBaseInfoResponseDTO.GroupInfoDTO();
        groupInfoDTO.setGroupNm("本地测试群-" + groupId);
        groupInfoDTO.setGroupOwnerId(EMPLOYEE_ID);
        groupInfoDTO.setGroupOwnerYstUserId("274658");
        groupInfoDTO.setGroupOwnerNm("本地客户经理");

        List<GroupBaseInfoResponseDTO.GroupMemberDTO> memberDTOList = new ArrayList<GroupBaseInfoResponseDTO.GroupMemberDTO>();
        memberDTOList.add(buildEmployeeMember(EMPLOYEE_ID, "本地客户经理"));
        memberDTOList.add(buildCustomerMember(CUSTOMER_ID, "本地客户"));
        groupInfoDTO.setGroupMemberList(memberDTOList);
        responseDTO.setInfBdy(Collections.singletonList(groupInfoDTO));
        return responseDTO;
    }

    @PostMapping("/customer/cust-base-info")
    public CustomerBaseInfoResponseDTO queryCustomerBaseInfo(@RequestBody final Map<String, Object> request) {
        String custId = valueOf(request.get("custId"));

        CustomerBaseInfoResponseDTO responseDTO = buildCustomerSuccess();
        if (!StringUtils.hasText(custId) || custId.contains("missing")) {
            CustomerBaseInfoResponseDTO.CustomerInfoDTO emptyCustomer = new CustomerBaseInfoResponseDTO.CustomerInfoDTO();
            responseDTO.setInfBdy(Collections.singletonList(emptyCustomer));
            return responseDTO;
        }

        CustomerBaseInfoResponseDTO.CustomerInfoDTO customerInfoDTO = new CustomerBaseInfoResponseDTO.CustomerInfoDTO();
        customerInfoDTO.setCustId(custId);
        customerInfoDTO.setCustNm("本地客户-" + custId);
        customerInfoDTO.setCustAvatar("http://mock/avatar/customer.png");

        CustomerBaseInfoResponseDTO.CustomerAuthDTO authDTO = new CustomerBaseInfoResponseDTO.CustomerAuthDTO();
        authDTO.setComId("1000000001");
        authDTO.setComUid("mock-company-uid");
        authDTO.setComNm("本地测试企业");
        authDTO.setComPosition("经办人");
        customerInfoDTO.setCustAuth(Collections.singletonList(authDTO));

        responseDTO.setInfBdy(Collections.singletonList(customerInfoDTO));
        return responseDTO;
    }

    @PostMapping("/employee/getUserInfoBySapId")
    public EmployeeInfoResponseDTO queryEmployeeInfo(@RequestBody final Map<String, Object> request) {
        String sapId = valueOf(request.get("sapId"));

        EmployeeInfoResponseDTO responseDTO = buildEmployeeSuccess();
        if (!StringUtils.hasText(sapId) || sapId.contains("missing")) {
            responseDTO.setInfBdy(Collections.singletonList(null));
            return responseDTO;
        }

        EmployeeInfoResponseDTO.EmployeeDTO employeeDTO = new EmployeeInfoResponseDTO.EmployeeDTO();
        employeeDTO.setSapId(sapId);
        employeeDTO.setId(sapId);
        employeeDTO.setName("本地员工-" + sapId);
        employeeDTO.setAlias("本地员工别名");
        employeeDTO.setGender("1");
        employeeDTO.setMobile("13800138000");
        employeeDTO.setTelephone("0755-12345678");
        employeeDTO.setAvatar("http://mock/avatar/employee.png");
        employeeDTO.setThumbAvatar("http://mock/avatar/employee-thumb.png");
        employeeDTO.setEmail("mock@example.com");
        employeeDTO.setPosition("客户经理");
        employeeDTO.setMainDept(1001);
        employeeDTO.setMainDeptNm("零售金融部");
        employeeDTO.setOpenUserid("mock-open-userid");
        employeeDTO.setCrmUserId("mock-crm-userid");
        employeeDTO.setBbkYstIntId("274658");
        employeeDTO.setQrCodeUrl("http://mock/qrcode/employee");
        employeeDTO.setCrtTm("2026-05-27 10:00:00");
        employeeDTO.setFollow("1");
        employeeDTO.setStatus("1");
        responseDTO.setInfBdy(Collections.singletonList(employeeDTO));
        return responseDTO;
    }

    private GroupBaseInfoResponseDTO buildGroupSuccess() {
        GroupBaseInfoResponseDTO responseDTO = new GroupBaseInfoResponseDTO();
        responseDTO.setRtnCod(SUCCESS_RTN_COD);
        responseDTO.setErrMsg(SUCCESS_ERR_MSG);
        GroupBaseInfoResponseDTO.ReturnInfoDTO returnInfoDTO = new GroupBaseInfoResponseDTO.ReturnInfoDTO();
        returnInfoDTO.setReturnCode("SUC0000");
        returnInfoDTO.setErrorMsg(SUCCESS_ERR_MSG);
        responseDTO.setReturnInfo(returnInfoDTO);
        return responseDTO;
    }

    private CustomerBaseInfoResponseDTO buildCustomerSuccess() {
        CustomerBaseInfoResponseDTO responseDTO = new CustomerBaseInfoResponseDTO();
        responseDTO.setRtnCod(SUCCESS_RTN_COD);
        responseDTO.setErrMsg(SUCCESS_ERR_MSG);
        CustomerBaseInfoResponseDTO.ReturnInfoDTO returnInfoDTO = new CustomerBaseInfoResponseDTO.ReturnInfoDTO();
        returnInfoDTO.setReturnCode("SUC0000");
        returnInfoDTO.setErrorMsg(SUCCESS_ERR_MSG);
        responseDTO.setReturnInfo(returnInfoDTO);
        return responseDTO;
    }

    private EmployeeInfoResponseDTO buildEmployeeSuccess() {
        EmployeeInfoResponseDTO responseDTO = new EmployeeInfoResponseDTO();
        responseDTO.setRtnCod(SUCCESS_RTN_COD);
        responseDTO.setErrMsg(SUCCESS_ERR_MSG);
        EmployeeInfoResponseDTO.ReturnInfoDTO returnInfoDTO = new EmployeeInfoResponseDTO.ReturnInfoDTO();
        returnInfoDTO.setReturnCode("SUC0000");
        returnInfoDTO.setErrorMsg(SUCCESS_ERR_MSG);
        responseDTO.setReturnInfo(returnInfoDTO);
        return responseDTO;
    }

    private GroupBaseInfoResponseDTO.GroupMemberDTO buildEmployeeMember(final String memberId, final String memberRemark) {
        GroupBaseInfoResponseDTO.GroupMemberDTO memberDTO = new GroupBaseInfoResponseDTO.GroupMemberDTO();
        memberDTO.setCustId(memberId);
        memberDTO.setCustNm(memberRemark);
        memberDTO.setUserType("1");
        memberDTO.setJoinTm("2026-05-27 10:00:00");
        memberDTO.setJoinScene("1");
        return memberDTO;
    }

    private GroupBaseInfoResponseDTO.GroupMemberDTO buildCustomerMember(final String memberId, final String memberRemark) {
        GroupBaseInfoResponseDTO.GroupMemberDTO memberDTO = new GroupBaseInfoResponseDTO.GroupMemberDTO();
        memberDTO.setCustId(memberId);
        memberDTO.setCustNm(memberRemark);
        memberDTO.setUserType("2");
        memberDTO.setJoinTm("2026-05-27 10:05:00");
        memberDTO.setJoinScene("1");
        return memberDTO;
    }

    private String valueOf(final Object value) {
        return value == null ? null : String.valueOf(value);
    }
}
