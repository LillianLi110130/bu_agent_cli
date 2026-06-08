CREATE TABLE `wecom_group` (
  `room_id` varchar(128) NOT NULL COMMENT '群编号',
  `group_name` varchar(256) DEFAULT NULL COMMENT '群名称',
  `group_owner_id` varchar(128) DEFAULT NULL COMMENT '群主企微ID',
  `group_owner_yst_user_id` varchar(128) DEFAULT NULL COMMENT '群主一事通ID',
  `group_owner_name` varchar(128) DEFAULT NULL COMMENT '群主在群中的姓名',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`room_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci COMMENT='企微群信息表';

CREATE TABLE `wecom_group_member` (
  `room_id` varchar(128) NOT NULL COMMENT '群编号',
  `member_id` varchar(128) NOT NULL COMMENT '成员企微ID',
  `member_remark` varchar(256) DEFAULT NULL COMMENT '群备注',
  `user_type` char(1) DEFAULT NULL COMMENT '成员类型，1员工，2客户',
  `join_time` varchar(64) DEFAULT NULL COMMENT '入群时间',
  `join_scene` varchar(64) DEFAULT NULL COMMENT '入群方式',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`room_id`, `member_id`),
  KEY `idx_member_id` (`member_id`),
  KEY `idx_user_type` (`user_type`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci COMMENT='企微群成员表';

CREATE TABLE `wecom_employee` (
  `employee_id` varchar(128) NOT NULL COMMENT '企微ID',
  `id` varchar(128) DEFAULT NULL COMMENT 'ID',
  `gender` varchar(16) DEFAULT NULL COMMENT '性别',
  `mobile` varchar(64) DEFAULT NULL COMMENT '手机号',
  `main_dept` int DEFAULT NULL COMMENT '主部门ID',
  `telephone` varchar(64) DEFAULT NULL COMMENT '电话',
  `avatar` varchar(512) DEFAULT NULL COMMENT '头像',
  `follow` varchar(64) DEFAULT NULL COMMENT '关注状态',
  `open_userid` varchar(128) DEFAULT NULL COMMENT '开放平台用户ID',
  `crm_user_id` varchar(128) DEFAULT NULL COMMENT 'CRM用户ID',
  `main_dept_nm` varchar(256) DEFAULT NULL COMMENT '主部门名称',
  `crt_tm` varchar(64) DEFAULT NULL COMMENT '创建时间',
  `qr_code_url` varchar(512) DEFAULT NULL COMMENT '二维码地址',
  `thumb_avatar` varchar(512) DEFAULT NULL COMMENT '缩略头像',
  `name` varchar(128) DEFAULT NULL COMMENT '姓名',
  `bbk_yst_int_id` varchar(128) DEFAULT NULL COMMENT '一事通ID',
  `alias` varchar(128) DEFAULT NULL COMMENT '别名',
  `position` varchar(256) DEFAULT NULL COMMENT '职位',
  `email` varchar(128) DEFAULT NULL COMMENT '邮箱',
  `status` varchar(64) DEFAULT NULL COMMENT '状态',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`employee_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci COMMENT='企微员工表';

CREATE TABLE `wecom_customer` (
  `cust_id` varchar(128) NOT NULL COMMENT '企微ID',
  `cust_name` varchar(128) DEFAULT NULL COMMENT '客户名称',
  `cust_avatar` varchar(512) DEFAULT NULL COMMENT '客户头像',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`cust_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci COMMENT='企微客户表';

CREATE TABLE `wecom_customer_auth` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键',
  `cust_id` varchar(128) NOT NULL COMMENT '企微ID',
  `com_id` varchar(128) DEFAULT NULL COMMENT '认证企业编号',
  `com_uid` varchar(128) DEFAULT NULL COMMENT '认证企业UID',
  `com_name` varchar(256) DEFAULT NULL COMMENT '认证企业名称',
  `com_position` varchar(128) DEFAULT NULL COMMENT '认证企业职务',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_cust_id` (`cust_id`),
  KEY `idx_com_id` (`com_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci COMMENT='客户认证企业表';

CREATE TABLE `sync_job_control` (
  `job_name` varchar(64) NOT NULL COMMENT '任务名',
  `lock_until` datetime DEFAULT NULL COMMENT '锁过期时间',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`job_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci COMMENT='定时任务控制表';

CREATE TABLE `sync_job_record` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键',
  `job_name` varchar(64) NOT NULL COMMENT '任务名',
  `start_time` datetime NOT NULL COMMENT '开始时间',
  `end_time` datetime DEFAULT NULL COMMENT '结束时间',
  `run_status` varchar(32) NOT NULL COMMENT '执行状态，RUNNING/SUCCESS/FAILED',
  `error_msg` varchar(512) DEFAULT NULL COMMENT '失败原因',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_job_name` (`job_name`),
  KEY `idx_start_time` (`start_time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci COMMENT='定时任务执行记录表';

insert into sync_job_control(job_name, lock_until) values ('wecom_group_sync', null);
insert into sync_job_control(job_name, lock_until) values ('wecom_customer_sync', null);
insert into sync_job_control(job_name, lock_until) values ('wecom_employee_sync', null);
