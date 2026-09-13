"""基于 SQL AST 的列级能力校验。"""
from sqlglot import exp, parse_one

from ..domain.exceptions import SqlPolicyError


def validate_columns(sql: str, capabilities: dict[str, dict[str, dict]]) -> str:
    """校验 SELECT 输出、过滤、排序、聚合使用的列能力。"""
    try:
        tree = parse_one(sql, read="mysql")
    except Exception as exc:
        raise SqlPolicyError("SQL 无法解析，已拒绝执行") from exc

    aliases = {}
    for table in tree.find_all(exp.Table):
        aliases[table.alias_or_name.lower()] = table.name.lower()
        aliases[table.name.lower()] = table.name.lower()

    def column_info(column, capability):
        table = (column.table or "").lower()
        name = column.name.lower()
        physical = aliases.get(table, table) if table else None
        if physical is None:
            matches = [(t, cols[name]) for t, cols in capabilities.items()
                       if name in cols]
            if not matches:
                raise SqlPolicyError(f"列 {column.name} 未登记或不允许使用")
            # 无前缀列只在所有候选列的能力一致时放行；敏感列仍一律拒绝。
            infos = [info for _, info in matches]
            if any(info.get("sensitive") for info in infos):
                raise SqlPolicyError(f"列 {column.name} 为敏感字段，拒绝查询")
            if not all(info.get(capability) for info in infos):
                raise SqlPolicyError(f"列 {column.name} 不允许用于 {capability}")
            return matches[0]
        cols = capabilities.get(physical)
        if cols is None or name not in cols:
            raise SqlPolicyError(f"列 {column.name} 未登记或不允许使用")
        return physical, cols[name]

    def check(columns, capability):
        for column in columns:
            if not isinstance(column, exp.Column):
                continue
            _, info = column_info(column, capability)
            if info.get("sensitive"):
                raise SqlPolicyError(f"列 {column.name} 为敏感字段，拒绝查询")
            if not info.get(capability):
                raise SqlPolicyError(f"列 {column.name} 不允许用于 {capability}")

    for projection in tree.find_all(exp.Select):
        for expression in projection.expressions:
            if isinstance(expression, exp.Star):
                raise SqlPolicyError("不允许使用 SELECT *，请明确指定可输出字段")
            check(expression.find_all(exp.Column), "selectable")

    for where in tree.find_all(exp.Where):
        check(where.find_all(exp.Column), "filterable")
    for order in tree.find_all(exp.Order):
        check(order.find_all(exp.Column), "sortable")
    for group in tree.find_all(exp.Group):
        check(group.find_all(exp.Column), "sortable")
    for aggregate in tree.find_all((exp.Sum, exp.Avg, exp.Min, exp.Max)):
        check(aggregate.find_all(exp.Column), "aggregatable")
    return sql
