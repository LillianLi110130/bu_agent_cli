package com.cmb.tg.tgai.service.wecom;

import com.cmb.tg.tgai.infrastructure.common.rest.wecom.client.WecomSyncClient;
import com.cmb.tg.tgai.infrastructure.common.rest.wecom.dto.response.CustomerBaseInfoResponseDTO;
import com.cmb.tg.tgai.infrastructure.common.rest.wecom.dto.response.EmployeeInfoResponseDTO;
import com.cmb.tg.tgai.infrastructure.common.rest.wecom.dto.response.GroupBaseInfoResponseDTO;
import com.cmb.tg.tgai.infrastructure.wecom.mapper.*;
import com.cmb.tg.tgai.infrastructure.wecom.po.*;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.util.CollectionUtils;
import org.springframework.util.StringUtils;

import java.time.LocalDateTime;
import java.util.*;

@Slf4j
@Service
@RequiredArgsConstructor
public class BaseInfoSyncService {

    private static final int ACTIVE_DAYS = 7;
    private static final String GROUP_JOB_NAME = "wecom_group_sync";
    private static final String CUSTOMER_JOB_NAME = "wecom_customer_sync";
    private static final String EMPLOYEE_JOB_NAME = "wecom_employee_sync";
    private static final String SUCCESS_RTN_COD = "0000200";

    private final SyncJobService syncJobService;
    private final WecomSyncClient wecomSyncClient;
    private final ChatMessageMapper chatMessageMapper;
    private final ChatGroupMapper chatGroupMapper;
    private final ChatGroupMemberMapper chatGroupMemberMapper;
    private final ChatCustomerMapper chatCustomerMapper;
    private final ChatEmployeeMapper chatEmployeeMapper;
    private final SyncPersistenceService syncPersistenceService;

    @Scheduled(cron = "${chat.sync.group.cron:0 0 2 * * ?}")
    public void syncGroupJob() {
        log.info("start sync group job");
        syncJobService.runWithControl(GROUP_JOB_NAME, () -> {
            doSyncGroup();
            return null;
        });
    }

    @Scheduled(cron = "${chat.sync.customer.cron:0 0 3 * * ?}")
    public void syncCustomerJob() {
        log.info("start sync customer job");
        syncJobService.runWithControl(CUSTOMER_JOB_NAME, () -> {
            doSyncCustomer();
            return null;
        });
    }

    @Scheduled(cron = "${chat.sync.employee.cron:0 20 3 * * ?}")
    public void syncEmployeeJob() {
        log.info("start sync employee job");
        syncJobService.runWithControl(EMPLOYEE_JOB_NAME, () -> {
            doSyncEmployee();
            return null;
        });
    }

    private void doSyncGroup() {
        LocalDateTime activeFromTime = LocalDateTime.now().minusDays(ACTIVE_DAYS);
        List<String> roomIds = chatMessageMapper.selectActiveRoomIds(activeFromTime);
        List<ChatGroupPO> groupPOList = new ArrayList<ChatGroupPO>();
        Map<String, List<ChatGroupMemberPO>> groupMemberMap = new LinkedHashMap<String, List<ChatGroupMemberPO>>();
        for (String roomId : roomIds) {
            GroupSyncData groupSyncData = queryGroupSyncData(roomId);
            if (groupSyncData == null) {
                continue;
            }
            groupPOList.add(groupSyncData.getGroupPO());
            groupMemberMap.put(roomId, groupSyncData.getMemberPOList());
        }
        if (!CollectionUtils.isEmpty(groupPOList)) {
            chatGroupMapper.batchUpsert(groupPOList);
        }
        for (Map.Entry<String, List<ChatGroupMemberPO>> entry : groupMemberMap.entrySet()) {
            try {
                syncPersistenceService.refreshGroupMembers(entry.getKey(), entry.getValue());
            } catch (Exception ex) {
                log.error("refresh group members failed, roomId={}", entry.getKey(), ex);
                throw ex;
            }
        }
        log.info("sync group job finished, groupCount={}", groupPOList.size());
    }

    private void doSyncCustomer() {
        LocalDateTime activeFromTime = LocalDateTime.now().minusDays(ACTIVE_DAYS);
        Set<String> customerIds = new HashSet<String>();
        customerIds.addAll(chatCustomerMapper.selectAllCustIds());
        customerIds.addAll(chatGroupMemberMapper.selectCustomerMemberIdsByActiveRooms(activeFromTime));
        customerIds.addAll(chatMessageMapper.selectSingleCustomerFromUserIds(activeFromTime));
        customerIds.addAll(chatMessageMapper.selectSingleReceiverIds(activeFromTime));
        List<ChatCustomerPO> customerPOList = new ArrayList<ChatCustomerPO>();
        Map<String, List<ChatCustomerAuthPO>> customerAuthMap =
                new LinkedHashMap<String, List<ChatCustomerAuthPO>>();

        for (String customerId : customerIds) {
            CustomerSyncData customerSyncData = queryCustomerSyncData(customerId);
            if (customerSyncData == null) {
                continue;
            }
            customerPOList.add(customerSyncData.getCustomerPO());
            customerAuthMap.put(customerSyncData.getCustId(), customerSyncData.getAuthPOList());
        }
        if (!CollectionUtils.isEmpty(customerPOList)) {
            chatCustomerMapper.batchUpsert(customerPOList);
        }
        for (Map.Entry<String, List<ChatCustomerAuthPO>> entry : customerAuthMap.entrySet()) {
            try {
                syncPersistenceService.refreshCustomerAuth(entry.getKey(), entry.getValue());
            } catch (Exception ex) {
                log.error("refresh customer auth failed, custId={}", entry.getKey(), ex);
                throw ex;
            }
        }
        log.info("sync customer job finished, customerCount={}", customerPOList.size());
    }

    private void doSyncEmployee() {
        LocalDateTime activeFromTime = LocalDateTime.now().minusDays(ACTIVE_DAYS);
        Set<String> employeeIds = new HashSet<String>();
        employeeIds.addAll(chatEmployeeMapper.selectAllEmployeeIds());
        employeeIds.addAll(chatGroupMemberMapper.selectEmployeeMemberIdsByActiveRooms(activeFromTime));
        employeeIds.addAll(chatMessageMapper.selectSingleEmployeeFromUserIds(activeFromTime));
        employeeIds.addAll(chatMessageMapper.selectSingleReceiverIds(activeFromTime));
        List<ChatEmployeePO> employeePOList = new ArrayList<ChatEmployeePO>();

        for (String employeeId : employeeIds) {
            ChatEmployeePO employeePO = queryEmployeePO(employeeId);
            if (employeePO == null) {
                continue;
            }
            employeePOList.add(employeePO);
        }
        if (!CollectionUtils.isEmpty(employeePOList)) {
            chatEmployeeMapper.batchUpsert(employeePOList);
        }
        log.info("sync employee job finished, employeeCount={}", employeePOList.size());
    }

    private GroupSyncData queryGroupSyncData(final String roomId) {
        GroupBaseInfoResponseDTO responseDTO;
        try {
            responseDTO = wecomSyncClient.queryGroupBaseInfo(roomId);
        } catch (Exception ex) {
            log.error("query group base info failed, roomId={}", roomId, ex);
            return null;
        }
        if (responseDTO == null || !SUCCESS_RTN_COD.equals(responseDTO.getRtnCod())) {
            log.error("query group base info not success, roomId={}, rtnCod={}, errMsg={}",
                    roomId,
                    responseDTO == null ? null : responseDTO.getRtnCod(),
                    responseDTO == null ? null : responseDTO.getErrMsg());
            return null;
        }
        if (CollectionUtils.isEmpty(responseDTO.getInfBdy())) {
            log.info("query group base info empty infBdy, roomId={}", roomId);
            return null;
        }
        GroupBaseInfoResponseDTO.GroupInfoDTO groupInfoDTO = responseDTO.getInfBdy().get(0);
        if (isEmptyGroupInfo(groupInfoDTO)) {
            log.info("query group base info empty, roomId={}", roomId);
            return null;
        }

        ChatGroupPO groupPO = new ChatGroupPO();
        groupPO.setRoomId(roomId);
        groupPO.setGroupName(groupInfoDTO.getGroupNm());
        groupPO.setGroupOwnerId(groupInfoDTO.getGroupOwnerId());
        groupPO.setGroupOwnerYstUserId(groupInfoDTO.getGroupOwnerYstUserId());
        groupPO.setGroupOwnerName(groupInfoDTO.getGroupOwnerNm());

        List<ChatGroupMemberPO> memberPOList = new ArrayList<ChatGroupMemberPO>();
        if (!CollectionUtils.isEmpty(groupInfoDTO.getGroupMemberList())) {
            for (GroupBaseInfoResponseDTO.GroupMemberDTO memberDTO : groupInfoDTO.getGroupMemberList()) {
                ChatGroupMemberPO memberPO = new ChatGroupMemberPO();
                memberPO.setRoomId(roomId);
                memberPO.setMemberId(memberDTO.getCustId());
                memberPO.setMemberRemark(memberDTO.getCustNm());
                memberPO.setUserType(memberDTO.getUserType());
                memberPO.setJoinTime(memberDTO.getJoinTm());
                memberPO.setJoinScene(memberDTO.getJoinScene());
                memberPOList.add(memberPO);
            }
        }
        return new GroupSyncData(groupPO, memberPOList);
    }

    private CustomerSyncData queryCustomerSyncData(final String customerId) {
        CustomerBaseInfoResponseDTO responseDTO;
        try {
            responseDTO = wecomSyncClient.queryCustomerBaseInfo(customerId);
        } catch (Exception ex) {
            log.error("query customer base info failed, custId={}", customerId, ex);
            return null;
        }
        if (responseDTO == null || !SUCCESS_RTN_COD.equals(responseDTO.getRtnCod())) {
            log.error("query customer base info not success, custId={}, rtnCod={}, errMsg={}",
                    customerId,
                    responseDTO == null ? null : responseDTO.getRtnCod(),
                    responseDTO == null ? null : responseDTO.getErrMsg());
            return null;
        }
        if (CollectionUtils.isEmpty(responseDTO.getInfBdy())) {
            log.info("query customer base info empty infBdy, custId={}", customerId);
            return null;
        }
        CustomerBaseInfoResponseDTO.CustomerInfoDTO customerInfoDTO = responseDTO.getInfBdy().get(0);
        if (customerInfoDTO == null || !StringUtils.hasText(customerInfoDTO.getCustId())) {
            log.info("query customer base info empty, custId={}", customerId);
            return null;
        }

        String custId = customerInfoDTO.getCustId();
        ChatCustomerPO customerPO = new ChatCustomerPO();
        customerPO.setCustId(custId);
        customerPO.setCustName(customerInfoDTO.getCustNm());
        customerPO.setCustAvatar(customerInfoDTO.getCustAvatar());
        return new CustomerSyncData(custId, customerPO, buildCustomerAuthPOList(custId, customerInfoDTO.getCustAuth()));
    }

    private ChatEmployeePO queryEmployeePO(final String employeeId) {
        EmployeeInfoResponseDTO responseDTO;
        try {
            responseDTO = wecomSyncClient.queryEmployeeInfo(employeeId);
        } catch (Exception ex) {
            log.error("query employee info failed, employeeId={}", employeeId, ex);
            return null;
        }
        if (responseDTO == null || !SUCCESS_RTN_COD.equals(responseDTO.getRtnCod())) {
            log.error("query employee info not success, employeeId={}, rtnCod={}, errMsg={}",
                    employeeId,
                    responseDTO == null ? null : responseDTO.getRtnCod(),
                    responseDTO == null ? null : responseDTO.getErrMsg());
            return null;
        }
        if (CollectionUtils.isEmpty(responseDTO.getInfBdy())) {
            log.info("query employee info empty infBdy, employeeId={}", employeeId);
            return null;
        }
        EmployeeInfoResponseDTO.EmployeeDTO employeeDTO = responseDTO.getInfBdy().get(0);
        if (employeeDTO == null || !StringUtils.hasText(employeeDTO.getSapId())) {
            log.info("query employee info empty, employeeId={}", employeeId);
            return null;
        }

        ChatEmployeePO employeePO = new ChatEmployeePO();
        employeePO.setEmployeeId(employeeDTO.getSapId());
        employeePO.setId(employeeDTO.getId());
        employeePO.setGender(employeeDTO.getGender());
        employeePO.setMobile(employeeDTO.getMobile());
        employeePO.setMainDept(employeeDTO.getMainDept());
        employeePO.setTelephone(employeeDTO.getTelephone());
        employeePO.setAvatar(employeeDTO.getAvatar());
        employeePO.setFollow(employeeDTO.getFollow());
        employeePO.setOpenUserid(employeeDTO.getOpenUserid());
        employeePO.setCrmUserId(employeeDTO.getCrmUserId());
        employeePO.setMainDeptNm(employeeDTO.getMainDeptNm());
        employeePO.setCrtTm(employeeDTO.getCrtTm());
        employeePO.setQrCodeUrl(employeeDTO.getQrCodeUrl());
        employeePO.setThumbAvatar(employeeDTO.getThumbAvatar());
        employeePO.setName(employeeDTO.getName());
        employeePO.setBbkYstIntId(employeeDTO.getBbkYstIntId());
        employeePO.setAlias(employeeDTO.getAlias());
        employeePO.setPosition(employeeDTO.getPosition());
        employeePO.setEmail(employeeDTO.getEmail());
        employeePO.setStatus(employeeDTO.getStatus());
        return employeePO;
    }

    private boolean isEmptyGroupInfo(final GroupBaseInfoResponseDTO.GroupInfoDTO groupInfoDTO) {
        if (groupInfoDTO == null) {
            return true;
        }
        return !StringUtils.hasText(groupInfoDTO.getGroupNm())
                && !StringUtils.hasText(groupInfoDTO.getGroupOwnerId())
                && !StringUtils.hasText(groupInfoDTO.getGroupOwnerYstUserId())
                && !StringUtils.hasText(groupInfoDTO.getGroupOwnerNm())
                && CollectionUtils.isEmpty(groupInfoDTO.getGroupMemberList());
    }

    private List<ChatCustomerAuthPO> buildCustomerAuthPOList(
            final String custId,
            final List<CustomerBaseInfoResponseDTO.CustomerAuthDTO> authDTOList) {
        List<ChatCustomerAuthPO> authPOList = new ArrayList<ChatCustomerAuthPO>();
        if (CollectionUtils.isEmpty(authDTOList)) {
            return authPOList;
        }
        for (CustomerBaseInfoResponseDTO.CustomerAuthDTO authDTO : authDTOList) {
            ChatCustomerAuthPO authPO = new ChatCustomerAuthPO();
            authPO.setCustId(custId);
            authPO.setComId(authDTO.getComId());
            authPO.setComUid(authDTO.getComUid());
            authPO.setComName(authDTO.getComNm());
            authPO.setComPosition(authDTO.getComPosition());
            authPOList.add(authPO);
        }
        return authPOList;
    }

    @Getter
    @RequiredArgsConstructor
    private static class GroupSyncData {
        private final ChatGroupPO groupPO;
        private final List<ChatGroupMemberPO> memberPOList;
    }

    @Getter
    @RequiredArgsConstructor
    private static class CustomerSyncData {
        private final String custId;
        private final ChatCustomerPO customerPO;
        private final List<ChatCustomerAuthPO> authPOList;
    }

}
