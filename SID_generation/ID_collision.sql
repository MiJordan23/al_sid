-- KNN防碰撞（对最后一级码本ID，此处为3级码本demo）
-- 随机防碰撞：sorted_index_lv3_col 的生成方式改为随机即可

-- 存储物品与代码簿的映射及索引信息（
CREATE TABLE IF NOT EXISTS item_codebook_info
(
    item_id          BIGINT
    ,origin_codebook STRING
    ,codebook        STRING
    ,index           BIGINT
)
LIFECYCLE 360
;

-- 从原始数据生成基础映射表
INSERT OVERWRITE TABLE item_codebook_info
SELECT  item_id
        ,codebook_index
        ,codebook_index
        ,ROW_NUMBER() OVER (PARTITION BY codebook_index ORDER BY rand() ) AS num
FROM    raw_data
;

-- 提取需要重新分配的物品ID
CREATE TABLE tmp_iid
LIFECYCLE 3 AS
SELECT  item_id
FROM    item_codebook_info
WHERE   index > 5
;

-- 截断保留前5个索引的数据
INSERT OVERWRITE TABLE item_codebook_info
SELECT  *
FROM    item_codebook_info
WHERE   index <= 5
;

-- 包含item和对应3级码本ID序的表 item_id, codebook_index, sorted_index, priority
INSERT OVERWRITE TABLE sorted_index_lv3
SELECT  *
FROM    sorted_index_lv3
LEFT ANTI JOIN  (
                    SELECT  item_id
                    FROM    item_codebook_info
                ) b
ON      a.item_id = b.item_id
;

-- 生成三级排序索引表
INSERT OVERWRITE TABLE sorted_index_lv3_col
SELECT  TRANS_ARRAY(2,',',a.item_id,codebook_index,SPLIT_PART(sorted_index,',',2,201),'1,2,3,...,200') AS (item_id,codebook_index,sorted_index,priority)
FROM    sorted_index_lv3 a
;

-- 过滤已满载的代码簿（保留未满5个物品的代码簿）
INSERT OVERWRITE TABLE sorted_index_lv3_col
SELECT  a.item_id
        ,a.codebook_index
        ,a.sorted_index
        ,a.priority
FROM    sorted_index_lv3_col a
LEFT ANTI JOIN  (
                    SELECT  item_id
                    FROM    item_codebook_info
                ) b
ON      a.item_id = b.item_id
LEFT ANTI JOIN  (
                    SELECT  codebook
                    FROM    (
                                SELECT  codebook
                                        ,COUNT(*) AS c
                                FROM    item_codebook_info
                                GROUP BY codebook
                            )
                    WHERE   c >= 5
                ) b
ON      a.sorted_index = b.codebook
;

-- 按优先级分配候选集
INSERT OVERWRITE TABLE candidate_assignment
SELECT  item_id
        ,codebook_index AS origin_cate
        ,sorted_index AS assigned_cate
        ,priority
FROM    (
            SELECT  item_id
                    ,codebook_index
                    ,sorted_index
                    ,priority
                    ,ROW_NUMBER() OVER (PARTITION BY codebook_index,priority ORDER BY rand(int(priority)) ) AS num
            FROM    sorted_index_lv3
        )
WHERE   num <= 5
;

-- 生成最终分配结果
INSERT OVERWRITE TABLE final_assignment
SELECT  assigned_cate
        ,origin_cate
        ,item_id
        ,priority
        ,rank
FROM    (
            SELECT  assigned_cate
                    ,origin_cate
                    ,BIGINT(item_id) AS item_id
                    ,BIGINT(priority) AS priority
                    ,ROW_NUMBER() OVER (PARTITION BY assigned_cate ORDER BY rank ) AS rank
            FROM    (
                        SELECT  assigned_cate
                                ,origin_cate
                                ,item_id
                                ,priority
                                ,5 + ROW_NUMBER() OVER (PARTITION BY assigned_cate ORDER BY priority,random(item_id) ) AS rank
                        FROM    candidate_assignment
                        UNION ALL
                        SELECT  codebook
                                ,origin_codebook
                                ,item_id
                                ,0 AS priority
                                ,BIGINT(index) AS rank
                        FROM    item_codebook_info
                    )
        )
WHERE   rank <= 5
;

-- 最终结果回写到主表
INSERT OVERWRITE TABLE item_codebook_info
SELECT  item_id
        ,origin_cate
        ,assigned_cate
        ,rank
FROM    (
            SELECT  item_id
                    ,origin_cate
                    ,assigned_cate
                    ,rank
                    ,ROW_NUMBER() OVER (PARTITION BY item_id ORDER BY priority ) AS rn
            FROM    final_assignment
        )
WHERE   rn = 1
;
-- 重复上述循环