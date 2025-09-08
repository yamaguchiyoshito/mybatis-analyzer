#!/usr/bin/env python3
"""
MyBatis Mapper分析ツール
MyBatisのMapperディレクトリを解析し、処理概要をMarkdownレポートとして出力する
"""

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import hashlib
import json
from datetime import datetime
import argparse
import sys
import csv
import copy
try:
    import sqlglot
    from sqlglot import parse_one
    SQLGLOT_AVAILABLE = True
except Exception:
    SQLGLOT_AVAILABLE = False


@dataclass(frozen=True)
class SQLStatement:
    """SQL文の情報を保持するイミュータブルクラス"""
    id: str
    type: str  # SELECT/INSERT/UPDATE/DELETE のいずれか
    content: str
    tables: Set[str]
    complexity: int
    has_dynamic: bool
    line_number: int


@dataclass(frozen=True)
class MapperEntity:
    """Mapperファイルの解析結果を保持"""
    file_path: str
    namespace: str
    statements: List[SQLStatement]
    dependencies: Set[str]
    result_maps: Dict[str, str]
    parameters: Set[str]
    includes: Set[str]
    sql_fragments: Dict[str, ET.Element] = field(default_factory=dict)


class MapperFileParser:
    """個別Mapperファイルの解析を担当"""
    
    def __init__(self):
        self.namespace_pattern = re.compile(r'namespace="([^"]+)"')
        self.table_pattern = re.compile(
            r'(?:FROM|JOIN|INTO|UPDATE|TABLE)\s+`?(\w+)`?', 
            re.IGNORECASE
        )
        self.param_pattern = re.compile(r'#\{([^}]+)\}')
        self.dynamic_tags = {'if', 'choose', 'when', 'otherwise', 'foreach', 'where', 'set'}
        
    def parse(self, file_path: Path) -> Optional[MapperEntity]:
        """ファイルタイプに応じた解析を実行"""
        if file_path.suffix == '.xml':
            return self._parse_xml_mapper(file_path)
        elif file_path.suffix == '.java':
            return self._parse_java_mapper(file_path)
        return None
    
    def _parse_xml_mapper(self, file_path: Path) -> Optional[MapperEntity]:
        """XMLマッパーファイルの解析"""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            namespace = root.get('namespace', '')
            statements = []
            dependencies = set()
            result_maps = {}
            includes = set()
            sql_fragments: Dict[str, ET.Element] = {}
            
            # resultMapの解析
            for result_map in root.findall('.//resultMap'):
                map_id = result_map.get('id', '')
                map_type = result_map.get('type', '')
                if map_id:
                    result_maps[map_id] = map_type
                extends = result_map.get('extends')
                if extends:
                    dependencies.add(extends)
            
            # SQL断片の解析 -> 先に収集しておく
            for sql_elem in root.findall('.//sql'):
                sql_id = sql_elem.get('id', '')
                if sql_id:
                    key = f"{namespace}.{sql_id}" if namespace else sql_id
                    # 要素を保存（deepcopyして後続のツリー操作による問題を回避）
                    sql_fragments[key] = copy.deepcopy(sql_elem)

            # SQL文の解析
            for stmt_type in ['select', 'insert', 'update', 'delete']:
                for elem in root.findall(f'.//{stmt_type}'):
                    stmt = self._extract_sql_statement(elem, stmt_type.upper(), sql_fragments, namespace)
                    if stmt:
                        statements.append(stmt)

                        # includeタグの検出
                        for include in elem.findall('.//include'):
                            refid = include.get('refid', '')
                            if refid:
                                includes.add(refid)
            
            # includes は既に収集済み
            
            # パラメータの抽出
            parameters = self._extract_parameters(ET.tostring(root, encoding='unicode'))
            
            return MapperEntity(
                file_path=str(file_path),
                namespace=namespace,
                statements=statements,
                dependencies=dependencies,
                result_maps=result_maps,
                parameters=parameters,
                includes=includes,
                sql_fragments=sql_fragments
            )
            
        except Exception as e:
            print(f"XMLパース エラー {file_path}: {e}", file=sys.stderr)
            return None
    
    def _parse_java_mapper(self, file_path: Path) -> Optional[MapperEntity]:
        """Javaインターフェースマッパーの解析"""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # パッケージ名の抽出
            package_match = re.search(r'package\s+([\w.]+);', content)
            namespace = package_match.group(1) if package_match else ''
            
            # インターフェース名の抽出
            interface_match = re.search(r'interface\s+(\w+)', content)
            if interface_match and namespace:
                namespace = f"{namespace}.{interface_match.group(1)}"
            
            statements = []
            
            # アノテーションベースのSQL文の抽出
            annotation_pattern = re.compile(
                r'@(Select|Insert|Update|Delete)\s*\(\s*["\']([^"\']+)["\']',
                re.IGNORECASE | re.DOTALL
            )
            
            for match in annotation_pattern.finditer(content):
                stmt_type = match.group(1).upper()
                sql_content = match.group(2)
                
                # 行番号の取得
                line_number = content[:match.start()].count('\n') + 1
                
                stmt = self._create_sql_statement(
                    id=f"method_{line_number}",
                    stmt_type=stmt_type,
                    content=sql_content,
                    line_number=line_number
                )
                statements.append(stmt)
            
            parameters = self._extract_parameters(content)
            
            return MapperEntity(
                file_path=str(file_path),
                namespace=namespace,
                statements=statements,
                dependencies=set(),
                result_maps={},
                parameters=parameters,
                includes=set()
            )
            
        except Exception as e:
            print(f"Javaパース エラー {file_path}: {e}", file=sys.stderr)
            return None
    
    def _extract_sql_statement(self, elem: ET.Element, stmt_type: str, sql_fragments: Dict[str, ET.Element], namespace: str) -> Optional[SQLStatement]:
        """XML要素からSQL文を抽出（include展開を含む）"""
        stmt_id = elem.get('id', '')
        if not stmt_id:
            return None
        # SQL文の内容を取得（include を再帰的に展開）
        sql_content = self._get_element_text(elem, sql_fragments=sql_fragments, namespace=namespace)

        # 動的SQLタグの存在確認（include 内の断片も含めて検査）
        has_dynamic = self._element_has_dynamic(elem, sql_fragments=sql_fragments, namespace=namespace)

        # 行番号の取得（簡易実装）
        line_number = 0

        return self._create_sql_statement(
            id=stmt_id,
            stmt_type=stmt_type,
            content=sql_content,
            line_number=line_number,
            has_dynamic=has_dynamic
        )
    
    def _create_sql_statement(self, id: str, stmt_type: str, content: str, 
                            line_number: int, has_dynamic: bool = False) -> SQLStatement:
        """SQLStatement オブジェクトの生成"""
        # テーブル名の抽出
        tables = set(self.table_pattern.findall(content))

        # 複雑度の計算
        complexity = self._calculate_complexity(content)

        return SQLStatement(
            id=id,
            type=stmt_type,
            content=content,  # 全文を保持
            tables=tables,
            complexity=complexity,
            has_dynamic=has_dynamic,
            line_number=line_number
        )
    
    def _get_element_text(self, elem: ET.Element, sql_fragments: Optional[Dict[str, ET.Element]] = None, namespace: str = '') -> str:
        """XML要素からテキストを再帰的に取得。

        変更: include 展開をサポートするため、sql_fragments と namespace を受け取り、
        <include refid="..."/> が見つかったら対応する断片を再帰的に展開して差し込む。
        互換性のため sql_fragments と namespace はオプション引数とする。
        """
        # 互換呼び出しをサポートするため柔軟なシグネチャ
        def _inner(e: ET.Element, sql_fragments: Optional[Dict[str, ET.Element]] = None, namespace: str = '') -> str:
            parts: List[str] = []
            if e.text:
                parts.append(e.text)
            for child in e:
                # include を見つけたら断片を展開
                if child.tag == 'include':
                    refid = child.get('refid', '')
                    if refid and sql_fragments is not None:
                        key = refid if '.' in refid else (f"{namespace}.{refid}" if namespace else refid)
                        frag_elem = sql_fragments.get(key) or sql_fragments.get(refid)
                        if frag_elem is not None:
                            parts.append(_inner(frag_elem, sql_fragments, namespace))
                        else:
                            if child.text:
                                parts.append(child.text)
                    else:
                        if child.text:
                            parts.append(child.text)
                else:
                    parts.append(_inner(child, sql_fragments, namespace))

                if child.tail:
                    parts.append(child.tail)
            # スペースで結合し、内部の空白を正規化
            return ' '.join(p for p in (pt.strip() for pt in parts) if p)

        # 元の互換呼び出しでは引数を取らないため、呼び出し時は既存コードとの互換性を保つ
        try:
            # 呼び出し側が拡張版シグネチャを使うことを想定して、内部に sql_fragments/namespace を渡す
            return _inner(elem, sql_fragments, namespace)
        except Exception:
            # フォールバック: 元の簡易実装
            text_parts = []
            if elem.text:
                text_parts.append(elem.text)
            for child in elem:
                # internal calls should pass through sql_fragments and namespace when available
                text_parts.append(self._get_element_text(child, sql_fragments=sql_fragments, namespace=namespace))
                if child.tail:
                    text_parts.append(child.tail)
            return ' '.join(text_parts)

    def _element_has_dynamic(self, elem: ET.Element, sql_fragments: Optional[Dict[str, ET.Element]] = None, namespace: str = '') -> bool:
        """要素内、及び include で差し込まれる断片内に動的タグが存在するかを再帰的に判定する。"""
        # 自分のタグが動的タグか
        if elem.tag in self.dynamic_tags:
            return True
        for child in elem:
            if child.tag == 'include':
                refid = child.get('refid', '')
                if refid and sql_fragments is not None:
                    key = refid if '.' in refid else (f"{namespace}.{refid}" if namespace else refid)
                    frag_elem = sql_fragments.get(key) or sql_fragments.get(refid)
                    if frag_elem is not None:
                        if self._element_has_dynamic(frag_elem, sql_fragments, namespace):
                            return True
                    # include の子要素自体もチェック
                    if self._element_has_dynamic(child, sql_fragments, namespace):
                        return True
                else:
                    if self._element_has_dynamic(child, sql_fragments, namespace):
                        return True
            else:
                if self._element_has_dynamic(child, sql_fragments, namespace):
                    return True
        return False
    
    def _extract_parameters(self, content: str) -> Set[str]:
        """パラメータの抽出"""
        return set(self.param_pattern.findall(content))
    
    def _calculate_complexity(self, sql: str) -> int:
        """SQL文の複雑度を計算"""
        complexity = 1
        
        # JOINの数
        complexity += len(re.findall(r'\bJOIN\b', sql, re.IGNORECASE))
        
        # サブクエリの数
        complexity += sql.count('(SELECT')
        
        # 条件分岐の数
        complexity += len(re.findall(r'\b(WHERE|AND|OR)\b', sql, re.IGNORECASE))
        
        # CASE文の数
        complexity += len(re.findall(r'\bCASE\b', sql, re.IGNORECASE))
        
        return complexity


class DirectoryScanner:
    """ディレクトリ構造の走査とファイル収集"""
    
    def __init__(self, exclude_patterns: Optional[List[str]] = None):
        self.exclude_patterns = exclude_patterns or ['target/', 'build/', '.git/']
        
    def scan(self, root_path: Path) -> List[Path]:
        """Mapperファイルの収集"""
        mapper_files = []
        
        for file_path in root_path.rglob('*'):
            # 除外パターンのチェック
            if any(pattern in str(file_path) for pattern in self.exclude_patterns):
                continue
                
            # Mapperファイルの判定
            if file_path.is_file():
                if file_path.suffix == '.xml' and 'mapper' in file_path.name.lower():
                    mapper_files.append(file_path)
                elif file_path.suffix == '.java' and 'mapper' in file_path.name.lower():
                    mapper_files.append(file_path)
        
        return sorted(mapper_files)


class SQLStatementAnalyzer:
    """SQL文の詳細解析と分類"""
    
    def __init__(self):
        self.crud_counter = Counter()
        self.table_access = defaultdict(set)
        self.complexity_scores = []
        
    def analyze(self, mappers: List[MapperEntity]) -> Dict[str, Any]:
        """全Mapperの統計分析"""
        total_statements = 0
        dynamic_count = 0
        table_crud = defaultdict(Counter)

        for mapper in mappers:
            for stmt in mapper.statements:
                total_statements += 1

                # CRUD操作のカウント
                self.crud_counter[stmt.type] += 1

                # テーブルアクセスパターンの記録
                for table in stmt.tables:
                    self.table_access[table].add(stmt.type)
                    # テーブルごとのCRUDカウント
                    table_crud[table][stmt.type] += 1

                # 複雑度スコアの記録
                self.complexity_scores.append(stmt.complexity)

                # 動的SQLのカウント
                if stmt.has_dynamic:
                    dynamic_count += 1

        # 統計情報の集計
        avg_complexity = sum(self.complexity_scores) / len(self.complexity_scores) if self.complexity_scores else 0

        return {
            'total_statements': total_statements,
            'crud_distribution': dict(self.crud_counter),
            'table_count': len(self.table_access),
            'table_list': list(self.table_access.keys()),
            'average_complexity': round(avg_complexity, 2),
            'max_complexity': max(self.complexity_scores) if self.complexity_scores else 0,
            'dynamic_sql_ratio': round(dynamic_count / total_statements * 100, 2) if total_statements > 0 else 0,
            'table_crud': {t: dict(c) for t, c in table_crud.items()}
        }


class DependencyGraph:
    """Mapper間の依存関係グラフ構築"""
    
    def __init__(self):
        self.graph = defaultdict(set)
        self.reverse_graph = defaultdict(set)
        
    def build(self, mappers: List[MapperEntity]) -> None:
        """依存関係グラフの構築"""
        namespace_map = {m.namespace: m for m in mappers}
        
        for mapper in mappers:
            # resultMap継承による依存
            for dep in mapper.dependencies:
                # 依存は名前空間付き (ns.id) または未修飾 id の可能性があるため両方を解決する
                if '.' in dep:
                    dep_namespace = dep.rsplit('.', 1)[0]
                    if dep_namespace in namespace_map:
                        self.graph[mapper.namespace].add(dep_namespace)
                        self.reverse_graph[dep_namespace].add(mapper.namespace)
                else:
                    # 名前空間でキー化された result_map をこの id で検索する
                    for ns, m in namespace_map.items():
                        if dep in m.result_maps:
                            self.graph[mapper.namespace].add(ns)
                            self.reverse_graph[ns].add(mapper.namespace)
                            break
            
            # include による依存
            for include_ref in mapper.includes:
                # include_ref は名前空間付き (ns.id) または未修飾 id の可能性がある
                if '.' in include_ref:
                    ref_namespace = include_ref.rsplit('.', 1)[0]
                    if ref_namespace in namespace_map:
                        self.graph[mapper.namespace].add(ref_namespace)
                        self.reverse_graph[ref_namespace].add(mapper.namespace)
                else:
                    # この id の SQL 断片を定義するマッパーを探す
                    for ns, m in namespace_map.items():
                        # MapperEntity.sql_fragments のキーは 'namespace.id' または 'id' として保存されている
                        qualified = f"{ns}.{include_ref}" if ns else include_ref
                        if qualified in m.sql_fragments or include_ref in m.sql_fragments:
                            self.graph[mapper.namespace].add(ns)
                            self.reverse_graph[ns].add(mapper.namespace)
                            break
    
    def find_cycles(self) -> List[List[str]]:
        """循環参照の検出"""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in rec_stack:
                    # 循環を検出
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])
            
            rec_stack.remove(node)
        
        for node in self.graph:
            if node not in visited:
                dfs(node, [])
        
        return cycles
    
    def get_metrics(self) -> Dict[str, Any]:
        """依存関係のメトリクス取得"""
        nodes = set(self.graph.keys()) | set(self.reverse_graph.keys())
        
        # 入次数と出次数の計算
        out_degrees = {node: len(self.graph[node]) for node in nodes}
        
        cycles = self.find_cycles()
        
    # レポート用の隣接マッピングを構築
        adjacency = {node: sorted(list(deps)) for node, deps in self.graph.items()}
        reverse_adjacency = {node: sorted(list(deps)) for node, deps in self.reverse_graph.items()}

        return {
            'total_nodes': len(nodes),
            'total_edges': sum(len(deps) for deps in self.graph.values()),
            'cycles_detected': len(cycles),
            'cycles': cycles[:5],  # 最初の5件のみ
            'average_dependencies': round(sum(out_degrees.values()) / len(nodes), 2) if nodes else 0,
            'adjacency': adjacency,
            'reverse_adjacency': reverse_adjacency
        }


class MetricsCalculator:
    """定量的メトリクスの算出"""
    
    def calculate(self, mappers: List[MapperEntity]) -> Dict[str, Any]:
        """各種メトリクスの計算"""
        metrics = {
            'file_count': len(mappers),
            'total_sql_statements': sum(len(m.statements) for m in mappers),
            'avg_statements_per_file': 0,
            'total_parameters': 0,
            'unique_parameters': set(),
            'namespace_count': len(set(m.namespace for m in mappers)),
            'result_map_count': sum(len(m.result_maps) for m in mappers),
            'duplicate_sql_hashes': []
        }
        
        # 平均SQL文数
        if metrics['file_count'] > 0:
            metrics['avg_statements_per_file'] = round(
                metrics['total_sql_statements'] / metrics['file_count'], 2
            )
        
        # パラメータ統計
        all_params = set()
        for mapper in mappers:
            all_params.update(mapper.parameters)
        metrics['unique_parameters'] = len(all_params)
        metrics['total_parameters'] = sum(len(m.parameters) for m in mappers)
        
        # 重複SQLの検出
        sql_hashes = defaultdict(list)
        for mapper in mappers:
            for stmt in mapper.statements:
                # SQL文の正規化とハッシュ化
                normalized = self._normalize_sql(stmt.content)
                hash_val = hashlib.md5(normalized.encode()).hexdigest()
                sql_hashes[hash_val].append((mapper.namespace, stmt.id))
        
        # 重複を検出
        duplicates = [
            {'hash': h, 'locations': locs} 
            for h, locs in sql_hashes.items() 
            if len(locs) > 1
        ]
        metrics['duplicate_sql_count'] = len(duplicates)
        
        return metrics
    
    def _normalize_sql(self, sql: str) -> str:
        """SQL文の正規化"""
        # 空白の正規化
        normalized = re.sub(r'\s+', ' ', sql)
        # パラメータの正規化
        normalized = re.sub(r'#\{[^}]+\}', '?', normalized)
        # 大文字小文字の統一
        normalized = normalized.upper()
        return normalized.strip()

    def _infer_table_map(self, mappers: List[MapperEntity]) -> Dict[str, str]:
        """簡易推論エンジン: マッパー内で出現するテーブルトークンを集め、
        同一ステートメント内で一緒に出現するトークン（正式名と思われる長いトークン）を
        キー->値マップとして推測する。

        戦略（軽量、ヒューリスティック）:
        - 各ステートメントの tables セットを収集
        - 各テーブルトークンの長さを比較し、短いトークン（略称）と長いトークン（正式名）が
          同一行に共起する場合、短い->長い のマッピング候補とする
        - 最も共起頻度が高いマッピングを採用する
        """
        cooccur = defaultdict(lambda: defaultdict(int))
        all_tokens = set()

        for m in mappers:
            for stmt in m.statements:
                toks = [t for t in stmt.tables if t]
                for t in toks:
                    all_tokens.add(t)
                # pairwise co-occurrence
                for a in toks:
                    for b in toks:
                        if a == b:
                            continue
                        cooccur[a][b] += 1

                # try to extract alias->table from SQL using sqlglot for higher accuracy
                if SQLGLOT_AVAILABLE and stmt.content:
                    try:
                        parsed = parse_one(stmt.content)
                        # walk parsed tree to find table nodes with alias
                        for node in parsed.walk():
                            cls = node.__class__.__name__.lower()
                            # table-like node
                            if 'table' in cls:
                                table_name = self._node_to_identifier(node)
                                alias_name = None
                                # sqlglot often stores alias in .args['alias'] or .alias
                                alias_node = None
                                if hasattr(node, 'alias') and node.alias:
                                    alias_node = node.alias
                                elif hasattr(node, 'args') and isinstance(node.args, dict):
                                    alias_node = node.args.get('alias')
                                if alias_node is not None:
                                    # alias_node may be an Alias expression
                                    alias_name = self._node_to_identifier(alias_node) or getattr(alias_node, 'this', None)
                                if alias_name and table_name:
                                    # record alias->table mapping directly
                                    all_tokens.add(alias_name)
                                    cooccur[alias_name][table_name] += 10  # boost weight for alias evidence
                    except Exception:
                        pass

                    # Fallback: regex-based alias extraction for common patterns when sqlglot not available
                    if stmt.content:
                        try:
                            s = stmt.content
                            # common patterns: FROM table alias, JOIN table alias, table AS alias
                            alias_patterns = [
                                r'\bFROM\s+([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)',
                                r'\bJOIN\s+([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)',
                                r'\bFROM\s+([A-Za-z_][A-Za-z0-9_\.]+)\s+AS\s+([A-Za-z_][A-Za-z0-9_]*)',
                                r'\bJOIN\s+([A-Za-z_][A-Za-z0-9_\.]+)\s+AS\s+([A-Za-z_][A-Za-z0-9_]*)',
                            ]
                            for pat in alias_patterns:
                                for mobj in re.finditer(pat, s, re.IGNORECASE):
                                    tbl = mobj.group(1)
                                    alias = mobj.group(2)
                                    if tbl and alias:
                                        # record alias and table tokens and boost cooccurrence
                                        all_tokens.add(alias)
                                        all_tokens.add(tbl)
                                        cooccur[alias][tbl] += 8
                        except Exception:
                            pass

        # 推論マップ構築: 各トークンについて、より長い（または長さ差がある）トークンに向けたマップを優先
        inferred: Dict[str, str] = {}
        for a in all_tokens:
            candidates = cooccur.get(a, {})
            if not candidates:
                continue
            # 候補をスコアでソート
            sorted_cands = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
            # 採用基準:
            # - 候補が十分に長い (len(b) >= len(a) + 2)
            # - またはキー自身が短い略称 (len(a) <= 3) の場合のみ採用
            chosen = None
            for b, score in sorted_cands:
                if len(b) >= len(a) + 2 or len(a) <= 3:
                    chosen = b
                    break
            if chosen is None:
                # 明確な長さ差がない場合は誤推論を避けるためスキップ
                continue
            inferred[a] = chosen

        # 返却: 小文字キーも含めておく
        out_map: Dict[str, str] = {}
        for k, v in inferred.items():
            out_map[k] = v
            out_map[k.lower()] = v

        return out_map

    # --- sqlglot 用 AST ヘルパー関数 ---
    def _node_to_identifier(self, node) -> Optional[str]:
        """sqlglot の AST ノードから識別子を抽出する補助関数。

        Identifier、Dot（schema.table）、ネストしたケースに対応し、
        抽出に失敗した場合は None を返します。
        """
        try:
            if node is None:
                return None

            # 直接的な name 属性
            name = getattr(node, 'name', None)
            if isinstance(name, str) and name:
                return name

            # sqlglot の一般的なパターン: node.this (Identifier または expression)
            val = getattr(node, 'this', None)
            if isinstance(val, str) and val:
                return val
            if hasattr(val, 'name'):
                v = getattr(val, 'name')
                if isinstance(v, str):
                    return v
            if hasattr(val, 'this'):
                return self._node_to_identifier(val)

            # Dot や修飾名: left/right や args を試す
            parts = []
            for attr in ('this', 'left', 'right', 'expression', 'this'):
                part = getattr(node, attr, None)
                if part is None:
                    continue
                p = self._node_to_identifier(part)
                if p:
                    parts.append(p)
            if parts:
                return '.'.join(parts)

            # フォールバック: 文字列解析
            s = str(node)
            m = re.search(r"([A-Za-z_][A-Za-z0-9_\.]+)", s)
            if m:
                return m.group(1)
        except Exception:
            return None
        return None

    def _extract_tables_and_functions_from_parsed(self, parsed) -> Tuple[Set[str], Set[str]]:
        """解析済み AST を走査してテーブル名と関数名を収集する。"""
        tables = set()
        funcs = set()
        try:
            for node in parsed.walk():
                cls = node.__class__.__name__.lower()
                # テーブルノード
                if 'table' in cls:
                    name = self._node_to_identifier(node)
                    if name:
                        tables.add(name)

                # 関数/呼び出しノード
                if 'func' in cls or 'function' in cls or cls.endswith('call') or 'anonymous' in cls:
                    fname = self._node_to_identifier(node)
                    if fname:
                        funcs.add(fname)
        except Exception:
            pass
        return tables, funcs

    def export_sql_metrics_csv(self, mappers: List[MapperEntity], csv_path: str, table_map: Optional[Dict[str, str]] = None) -> None:
        """各SQL文ごとのメトリクスをCSVに出力する

        出力するカラム:
        namespace, file, statement_id, type, line_count, element_count, complexity, has_dynamic
        """
        # 要素数をカウントするヘルパー関数（キーワードまたはASTノードのフォールバック）
        def count_elements_local(sql: str) -> int:
            if SQLGLOT_AVAILABLE:
                try:
                    tree = parse_one(sql)
                    return max(1, sum(1 for _ in tree.walk()))
                except Exception:
                    pass
            kw_patterns = [r'\bSELECT\b', r'\bINSERT\b', r'\bUPDATE\b', r'\bDELETE\b', r'\bFROM\b', r'\bJOIN\b']
            cnt = 0
            for pat in kw_patterns:
                cnt += len(re.findall(pat, sql, re.IGNORECASE))
            return max(1, cnt)

        # 最初に行を収集してから分位数とクラスタを計算
        rows: List[Dict[str, Any]] = []
        for mapper in mappers:
            for stmt in mapper.statements:
                content = stmt.content or ''

                # 前処理: XMLタグっぽい部分を除去し、動的プレースホルダを正規化
                cleaned = re.sub(r'<[^>]+>', ' ', content)
                cleaned = re.sub(r'#\{[^}]+\}', '?', cleaned)
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()

                # 行数
                line_count = content.count('\n') + 1 if content else 0

                element_count = count_elements_local(cleaned)

                normalized = self._normalize_sql(cleaned)
                sql_hash = hashlib.md5(normalized.encode()).hexdigest()

                # テーブルを抽出（可能であればsqlglotを使用）し、エイリアスを解決
                tables_list = ''
                tables_detail = ''
                if SQLGLOT_AVAILABLE:
                    try:
                        parsed = parse_one(cleaned)
                        tbls, funcs = self._extract_tables_and_functions_from_parsed(parsed)
                        # ローカルエイリアス->テーブルマッピングと実際のテーブル名のセットを構築
                        local_alias_map = {}
                        real_tables = set()
                        for node in parsed.walk():
                            cls = node.__class__.__name__.lower()
                            if 'table' in cls:
                                tname = self._node_to_identifier(node)
                                alias_node = None
                                if hasattr(node, 'alias') and node.alias:
                                    alias_node = node.alias
                                elif hasattr(node, 'args') and isinstance(node.args, dict):
                                    alias_node = node.args.get('alias')
                                aname = None
                                if alias_node is not None:
                                    aname = self._node_to_identifier(alias_node) or getattr(alias_node, 'this', None)
                                if tname:
                                    real_tables.add(tname)
                                if aname and tname:
                                    local_alias_map[aname] = tname

                        # real_tablesが見つかった場合はそれを優先、それ以外はtblsを使用
                        if real_tables:
                            tables_list = ','.join(sorted(real_tables))
                        elif tbls:
                            tables_list = ','.join(sorted(tbls))
                        else:
                            tables_list = ''

                        tables_detail = tables_list
                        functions_detail = ','.join(sorted(funcs)) if funcs else ''
                    except Exception:
                        tables_list = ','.join(sorted(stmt.tables)) if stmt.tables else ''
                        tables_detail = tables_list
                        functions_detail = ''
                else:
                    # フォールバック（sqlglotなし）: 正規表現によるエイリアス検出と実際のテーブル名の優先
                    s2 = cleaned
                    local_alias_map = {}
                    try:
                        for mobj in re.finditer(r"\\bFROM\\s+([A-Za-z_][A-Za-z0-9_\\.]*)\\s+(?:AS\\s+)?([A-Za-z_][A-Za-z0-9_]*)", s2, re.IGNORECASE):
                            tbl = mobj.group(1)
                            alias = mobj.group(2)
                            if tbl and alias:
                                local_alias_map[alias] = tbl
                        for mobj in re.finditer(r"\\bJOIN\\s+([A-Za-z_][A-Za-z0-9_\\.]*)\\s+(?:AS\\s+)?([A-Za-z_][A-Za-z0-9_]*)", s2, re.IGNORECASE):
                            tbl = mobj.group(1)
                            alias = mobj.group(2)
                            if tbl and alias:
                                local_alias_map[alias] = tbl
                    except Exception:
                        local_alias_map = {}

                    # 実際のテーブル名のセットを構築: マッピングされたターゲットとstmt.tablesからの任意のテーブル
                    real_tables = set()
                    for alias, tbl in local_alias_map.items():
                        real_tables.add(tbl)
                    for t in stmt.tables:
                        if t and t not in local_alias_map.keys():
                            real_tables.add(t)

                    if real_tables:
                        tables_list = ','.join(sorted(real_tables))
                    else:
                        tables_list = ','.join(sorted(stmt.tables)) if stmt.tables else ''
                    tables_detail = tables_list
                    functions_detail = ''

                rows.append({
                    'namespace': mapper.namespace,
                    'file': Path(mapper.file_path).name,
                    'statement_id': stmt.id,
                    'type': stmt.type,
                    'line_count': line_count,
                    'element_count': element_count,
                    'complexity': stmt.complexity,
                    'has_dynamic': int(stmt.has_dynamic),
                    'sql_hash': sql_hash,
                    'tables': tables_list,
                    'param_count': len(re.findall(r'#\{[^}]+\}', content)),
                    'tables_detail': tables_detail,
                    'functions_detail': functions_detail
                })

        # 標準的なヒューリスティックとデータセットの分位数を使用して閾値を計算
        complexities = sorted([r['complexity'] for r in rows])
        locs = sorted([r['line_count'] for r in rows])

        def percentile(sorted_list: List[int], p: float) -> float:
            if not sorted_list:
                return 0.0
            k = (len(sorted_list)-1) * p
            f = int(k)
            c = min(f+1, len(sorted_list)-1)
            if f == c:
                return float(sorted_list[int(k)])
            d0 = sorted_list[f] * (c - k)
            d1 = sorted_list[c] * (k - f)
            return (d0 + d1)

        p33_c = percentile(complexities, 0.33)
        p66_c = percentile(complexities, 0.66)
        p33_l = percentile(locs, 0.33)
        p66_l = percentile(locs, 0.66)

        # 標準的なヒューリスティック（参考: 一般的なコード品質指標に基づく閾値）
        std_c1, std_c2 = 5, 15
        std_l1, std_l2 = 10, 50

        thr_c1 = max(std_c1, p33_c)
        thr_c2 = max(std_c2, p66_c)
        thr_l1 = max(std_l1, p33_l)
        thr_l2 = max(std_l2, p66_l)

        def cat(v: float, t1: float, t2: float) -> str:
            if v <= t1:
                return 'Low'
            if v <= t2:
                return 'Medium'
            return 'High'

        # クラスタラベルを割り当て
        for r in rows:
            ccat = cat(r['complexity'], thr_c1, thr_c2)
            lcat = cat(r['line_count'], thr_l1, thr_l2)
            # 基本クラスタ名（例: 'Low/Medium'）
            base_cluster = f'{ccat}/{lcat}'
            r['cluster'] = base_cluster

        # クラスタにソート可能な数値コード01..09をプレフィックス（行優先順: Low/Low..High/High）
        cluster_order = [
            'Low/Low','Low/Medium','Low/High',
            'Medium/Low','Medium/Medium','Medium/High',
            'High/Low','High/Medium','High/High'
        ]
        cluster_code = {name: f"{i:02d}" for i, name in enumerate(cluster_order, start=1)}
        for r in rows:
            base = r.get('cluster', '')
            code = cluster_code.get(base, '00')
            r['cluster'] = f"{code}_{base}"

        # 必要に応じてテーブルマッピング推論を適用（前述のロジックを使用）
        if table_map is None:
            table_map = self._infer_table_map(mappers)
            # デバッグ: 推論されたマッピングを表示
            try:
                print(f"[DEBUG] 推論されたテーブルマップ: {json.dumps(table_map, ensure_ascii=False) }", file=sys.stderr)
            except Exception:
                print(f"[DEBUG] 推論されたテーブルマップ: {table_map}", file=sys.stderr)

        for r in rows:
            if table_map and r.get('tables'):
                try:
                    mapped_tokens = []
                    seen = set()
                    for t in [x.strip() for x in r['tables'].split(',') if x.strip()]:
                        mapped = table_map.get(t) or table_map.get(t.lower())
                        out = mapped if mapped else t
                        if out not in seen:
                            mapped_tokens.append(out)
                            seen.add(out)
                    r['tables'] = ','.join(mapped_tokens)

                    mapped_detail_tokens = []
                    seen_d = set()
                    for t in [x.strip() for x in r['tables_detail'].split(',') if x.strip()]:
                        mapped = table_map.get(t) or table_map.get(t.lower())
                        out = mapped if mapped else t
                        if out not in seen_d:
                            mapped_detail_tokens.append(out)
                            seen_d.add(out)
                    r['tables_detail'] = ','.join(mapped_detail_tokens)
                except Exception:
                    pass

        # 新しいクラスタ列を含むCSVを書き出し
        try:
            with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['namespace', 'file', 'statement_id', 'type', 'line_count', 'element_count', 'complexity', 'has_dynamic', 'sql_hash', 'tables', 'param_count', 'cluster', 'tables_detail', 'functions_detail'])
                for r in rows:
                    writer.writerow([
                        r['namespace'],
                        r['file'],
                        r['statement_id'],
                        r['type'],
                        r['line_count'],
                        r['element_count'],
                        r['complexity'],
                        r['has_dynamic'],
                        r['sql_hash'],
                        r['tables'],
                        r['param_count'],
                        r['cluster'],
                        r.get('tables_detail', ''),
                        r.get('functions_detail', '')
                    ])
        except Exception as e:
            print(f"CSV 出力エラー: {e}", file=sys.stderr)

        # 複雑度と行数の散布図を生成し、クラスタごとに色分け
        try:
            import matplotlib.pyplot as plt
            out_img = str(Path(csv_path).with_name(Path(csv_path).stem + '_complexity_loc.png'))
            # クラスタから色へのマッピングを準備
            cluster_vals = sorted(list({r['cluster'] for r in rows}))
            colors = plt.get_cmap('tab10')
            color_map = {v: colors(i % 10) for i, v in enumerate(cluster_vals)}

            xs = [r['line_count'] for r in rows]
            ys = [r['complexity'] for r in rows]
            cs = [color_map[r['cluster']] for r in rows]

            plt.figure(figsize=(8,6))
            for v in cluster_vals:
                vx = [r['line_count'] for r in rows if r['cluster']==v]
                vy = [r['complexity'] for r in rows if r['cluster']==v]
                plt.scatter(vx, vy, label=v, alpha=0.7)

            # 閾値線を描画
            plt.axvline(thr_l1, color='gray', linestyle='--', linewidth=0.8)
            plt.axvline(thr_l2, color='gray', linestyle='--', linewidth=0.8)
            plt.axhline(thr_c1, color='gray', linestyle='--', linewidth=0.8)
            plt.axhline(thr_c2, color='gray', linestyle='--', linewidth=0.8)

            plt.xlabel('コード行数 (LOC)')
            plt.ylabel('SQLの複雑度')
            plt.title('SQLの複雑度 vs LOC (3x3クラスタ)')
            plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(out_img)
            plt.close()
            # また、CSVの隣にクラスタカウントマトリクスをJSONとして書き出す
            try:
                from collections import Counter as _Counter
                cluster_counts = _Counter([r['cluster'] for r in rows])
                # プレフィックス付きキーを正規化（例: '01_Low/Low' -> 'Low/Low'）
                counts_base = {}
                for k, v in cluster_counts.items():
                    if isinstance(k, str) and '_' in k:
                        base = k.split('_', 1)[1]
                    else:
                        base = k
                    counts_base[base] = counts_base.get(base, 0) + v

                # 正規化されたベースから3x3マトリクスを構築
                matrix = {
                    'Low/Low': counts_base.get('Low/Low', 0),
                    'Low/Medium': counts_base.get('Low/Medium', 0),
                    'Low/High': counts_base.get('Low/High', 0),
                    'Medium/Low': counts_base.get('Medium/Low', 0),
                    'Medium/Medium': counts_base.get('Medium/Medium', 0),
                    'Medium/High': counts_base.get('Medium/High', 0),
                    'High/Low': counts_base.get('High/Low', 0),
                    'High/Medium': counts_base.get('High/Medium', 0),
                    'High/High': counts_base.get('High/High', 0)
                }
                json_out = str(Path(csv_path).with_name(Path(csv_path).stem + '_clusters.json'))
                with open(json_out, 'w', encoding='utf-8') as jf:
                    json.dump({'matrix': matrix, 'counts_prefixed': dict(cluster_counts), 'counts_base': counts_base}, jf, ensure_ascii=False, indent=2)
            except Exception:
                pass
        except Exception as e:
            print(f"プロット作成エラー（matplotlib が必要）: {e}", file=sys.stderr)
        


class ReportGenerator:
    """Markdown形式のレポート生成"""
    
    def __init__(self):
        self.report_sections = []
        
    def generate(self, 
                mappers: List[MapperEntity],
                sql_analysis: Dict[str, Any],
                dependency_metrics: Dict[str, Any],
                general_metrics: Dict[str, Any],
                output_path: str) -> None:
        """完全なレポートの生成"""
        
        # ヘッダー
        self._add_header()
        
        # サマリー
        self._add_summary(general_metrics, sql_analysis, dependency_metrics)
        
        # ファイル統計
        self._add_file_statistics(general_metrics)
        
        # SQL文統計
        self._add_sql_statistics(general_metrics, sql_analysis)
        
        # 依存関係
        self._add_dependency_analysis(dependency_metrics)
        
        # テーブル一覧
        self._add_table_list(sql_analysis)

        # Mapper詳細
        self._add_mapper_details(mappers)

        # クラスタ分布（CSV に隣接する JSON とプロット画像を埋め込む）
        self._add_cluster_section(output_path)

        # ファイルへの出力
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(self.report_sections))

    def _add_cluster_section(self, output_path: str) -> None:
        """CSV により生成されたクラスタ分布図と 3x3 マトリクスをレポートに追加する。

        期待する配置:
        - output/mapper_sql_metrics_complexity_loc.png を図として埋め込み（相対パス）
        - output/mapper_sql_metrics_clusters.json を読み、3x3 テーブルを出力
        """
        # レポート出力ディレクトリを基準にファイルを検索
        out_dir = Path(output_path).parent
        img_path = out_dir / 'mapper_sql_metrics_complexity_loc.png'
        json_path = out_dir / 'mapper_sql_metrics_clusters.json'

        section_lines = []
        section_lines.append('## クラスタ分布')
        section_lines.append('')
        # 画像が存在する場合は埋め込む
        if img_path.exists():
            # レポートと画像が同じフォルダにある場合に画像参照が機能するように、同一ディレクトリの相対パスで埋め込む
            section_lines.append(f'![SQL complexity vs LOC](./{img_path.name})')
            section_lines.append('')

        # JSON ファイルが存在する場合は読み込む
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as jf:
                    d = json.load(jf)
                    matrix = d.get('matrix', {})

                section_lines.append('### クラスタ件数マトリクス')
                section_lines.append('')
                # 各行の合計と割合を表示する列を追加
                total = sum(int(v) for v in matrix.values()) if matrix else 0
                section_lines.append('| Complexity \\ LOC | Low | Medium | High | 合計 (割合) |')
                section_lines.append('|---|---:|---:|---:|---:|')
                # 行をフォーマットするヘルパー関数
                def _row_line(a, b, c, label):
                    row_total = int(a) + int(b) + int(c)
                    pct = (row_total / total * 100) if total > 0 else 0.0
                    return f"| {label} | {a} | {b} | {c} | {row_total} ({pct:.1f}%) |"

                section_lines.append(_row_line(matrix.get('Low/Low',0), matrix.get('Low/Medium',0), matrix.get('Low/High',0), 'Low'))
                section_lines.append(_row_line(matrix.get('Medium/Low',0), matrix.get('Medium/Medium',0), matrix.get('Medium/High',0), 'Medium'))
                section_lines.append(_row_line(matrix.get('High/Low',0), matrix.get('High/Medium',0), matrix.get('High/High',0), 'High'))
            except Exception:
                section_lines.append('*クラスタ情報の読み込みに失敗しました*')
        else:
            section_lines.append('*クラスタ出力（mapper_sql_metrics_clusters.json または画像）が見つかりません。CSV を生成してから再実行してください。*')

        self.report_sections.append('\n'.join(section_lines))
    
    def _add_header(self) -> None:
        """レポートヘッダーの追加"""
        header = f"""# MyBatis Mapper分析レポート
生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---"""
        self.report_sections.append(header)
    
    def _add_summary(self, general: Dict, sql: Dict, deps: Dict) -> None:
        """サマリーの追加"""
        summary = f"""## サマリー

| 項目 | 値 |
|-----|-----|
| 総Mapperファイル数 | {general['file_count']} |
| 総SQL文数 | {general['total_sql_statements']} |
| 平均SQL文数/ファイル | {general['avg_statements_per_file']} |
| 平均複雑度 | {sql.get('average_complexity', 0)} |
| 最大複雑度 | {sql.get('max_complexity', 0)} |
| 動的SQL使用率 | {sql.get('dynamic_sql_ratio', 0)}% |
| 循環参照 | {deps['cycles_detected']}件 |
| 重複SQL | {general.get('duplicate_sql_count', 0)}件 |"""
        
        self.report_sections.append(summary)
    
    def _add_file_statistics(self, general: Dict) -> None:
        """ファイル統計の追加"""
        stats = f"""## ファイル統計

| メトリクス | 値 |
|----------|-----|
| Mapperファイル数 | {general['file_count']} |
| 名前空間数 | {general['namespace_count']} |
| ResultMap定義数 | {general['result_map_count']} |
| ユニークパラメータ数 | {general['unique_parameters']} |
| 総パラメータ数 | {general['total_parameters']} |"""
        
        self.report_sections.append(stats)
    
    def _add_sql_statistics(self, general: Dict, sql: Dict) -> None:
        """SQL文統計の追加"""
        stats = f"""## SQL文統計

### 操作タイプ別分布
| 操作タイプ | 件数 | 割合 |
|----------|------|------|"""
        
        crud = sql.get('crud_distribution', {})
        total = general['total_sql_statements']
        for op_type in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']:
            count = crud.get(op_type, 0)
            percentage = (count / total * 100) if total > 0 else 0
            stats += f"\n| {op_type} | {count} | {percentage:.1f}% |"
        
        stats += f"""

### 複雑度統計
| メトリクス | 値 |
|----------|-----|
| 平均複雑度 | {sql.get('average_complexity', 0)} |
| 最大複雑度 | {sql.get('max_complexity', 0)} |
| 動的SQL使用率 | {sql.get('dynamic_sql_ratio', 0)}% |"""
        
        self.report_sections.append(stats)
    
    def _add_dependency_analysis(self, deps: Dict) -> None:
        """依存関係分析の追加"""
        section = f"""## 依存関係分析

| メトリクス | 値 |
|----------|-----|
| 総ノード数 | {deps['total_nodes']} |
| 総エッジ数 | {deps['total_edges']} |
| 平均依存数 | {deps['average_dependencies']} |
| 循環参照数 | {deps['cycles_detected']} |"""
        
        # 循環参照がある場合は詳細を表示
        if deps['cycles_detected'] > 0 and deps.get('cycles'):
            section += "\n\n### 検出された循環参照"
            for i, cycle in enumerate(deps['cycles'][:5], 1):
                cycle_str = ' → '.join(cycle)
                section += f"\n{i}. {cycle_str}"
        # NOTE: 隣接リストの詳細は別ファイルに出力するためここでは追加しない
        self.report_sections.append(section)

    def generate_adjacency_report(self, dependency_metrics: Dict[str, Any], output_path: str, graph_path: Optional[str] = None) -> None:
        """依存関係の隣接リストを別ファイルで出力し、Graphviz画像も生成する

        - output_path: Markdownファイルの出力先
        - graph_path: 画像出力先（例: output/mapper_graph.png）。Noneまたは空文字で画像出力をスキップ
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        adjacency = dependency_metrics.get('adjacency', {})
        reverse_adj = dependency_metrics.get('reverse_adjacency', {})

        lines: List[str] = []
        lines.append('# Mapper 呼び出し関係レポート')
        lines.append(f'生成日時: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        lines.append('')
        lines.append('## 概要')
        lines.append('')
        lines.append('| メトリクス | 値 |')
        lines.append('|---|---:|')
        lines.append(f"| 総ノード数 | {dependency_metrics.get('total_nodes', 0)} |")
        lines.append(f"| 総エッジ数 | {dependency_metrics.get('total_edges', 0)} |")
        lines.append(f"| 平均依存数 | {dependency_metrics.get('average_dependencies', 0)} |")
        lines.append(f"| 循環参照数 | {dependency_metrics.get('cycles_detected', 0)} |")

        # 隣接リストテーブル
        lines.append('')
        lines.append('## 隣接リスト（呼び出し先 / 呼び出し元）')
        lines.append('')
        lines.append('| Mapper (namespace) | 呼び出し先 (outgoing) | 呼び出し元 (incoming) |')
        lines.append('|---|---|---|')
        all_namespaces = sorted(set(list(adjacency.keys()) + list(reverse_adj.keys())))
        for ns in all_namespaces:
            outs = ', '.join(adjacency.get(ns, [])) or '-'
            ins = ', '.join(reverse_adj.get(ns, [])) or '-'
            lines.append(f'| {ns} | {outs} | {ins} |')

        # ファイル書き込み (Markdown)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
        except Exception as e:
            print(f"隣接レポート出力エラー: {e}", file=sys.stderr)

        # JSON 出力（同じベース名で .json を出す）
        try:
            json_path = str(Path(output_path).with_suffix('.json'))
            with open(json_path, 'w', encoding='utf-8') as jf:
                json.dump({'adjacency': adjacency, 'reverse_adjacency': reverse_adj, 'metrics': dependency_metrics}, jf, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"隣接JSON出力エラー: {e}", file=sys.stderr)

        # Graphviz 出力（オプション）
        if graph_path:
            try:
                # Build adjacency graph components (connected components on undirected version)
                nodes = set(adjacency.keys()) | set(reverse_adj.keys())
                for dests in adjacency.values():
                    nodes.update(dests)

                # Build undirected adjacency for components
                undirected = defaultdict(set)
                for src, dests in adjacency.items():
                    for d in dests:
                        undirected[src].add(d)
                        undirected[d].add(src)

                # find connected components
                components = []
                visited = set()
                for n in nodes:
                    if n in visited:
                        continue
                    stack = [n]
                    comp = set()
                    while stack:
                        cur = stack.pop()
                        if cur in visited:
                            continue
                        visited.add(cur)
                        comp.add(cur)
                        for nb in undirected.get(cur, []):
                            if nb not in visited:
                                stack.append(nb)
                    components.append(sorted(comp))

                # generate one graph per component (or a single file if only one)
                base_graph_path = Path(graph_path)
                Path(base_graph_path.parent).mkdir(parents=True, exist_ok=True)

                import graphviz
                for idx, comp in enumerate(components, start=1):
                    dot = graphviz.Digraph(format='png')
                    dot.attr(rankdir='LR')
                    # スタイリングのための次数を計算
                    deg_in = {n: 0 for n in comp}
                    deg_out = {n: 0 for n in comp}
                    for src in comp:
                        for dst in adjacency.get(src, []):
                            if dst in comp:
                                deg_out[src] += 1
                                deg_in[dst] += 1

                    # add nodes with styling:
                    for n in comp:
                        ino = deg_in.get(n, 0)
                        outo = deg_out.get(n, 0)
                        # self-loop only (孤立した自己参照)
                        if ino == 1 and outo == 1 and adjacency.get(n, []) == [n]:
                            dot.node(n, label=n, color='gray80', style='filled', fillcolor='gray90', fontsize='8')
                        # 孤立ノード（入出力なし）
                        elif ino == 0 and outo == 0:
                            dot.node(n, label=n, color='gray70', style='filled', fillcolor='gray85', fontsize='8')
                        else:
                            # 外部エッジを持つノードをハイライト
                            # 次数に応じてサイズを変更
                            degree = ino + outo
                            size = min(2.5, 0.8 + degree * 0.3)
                            color = 'lightblue' if outo > 0 else 'lightgreen'
                            dot.node(n, label=n, color='black', style='filled', fillcolor=color, fontsize=str(10 + degree), width=str(size))

                    # add edges within component
                    for src in comp:
                        for dst in adjacency.get(src, []):
                            if dst in comp:
                                # thin edge for self-loop, bold for cross-node
                                if src == dst:
                                    dot.edge(src, dst, penwidth='0.5', color='gray60')
                                else:
                                    dot.edge(src, dst, penwidth='1.5', color='black')
                    # choose filename
                    if len(components) == 1:
                        out_path = str(base_graph_path)
                    else:
                        out_path = str(base_graph_path.with_name(base_graph_path.stem + f'_{idx}').with_suffix(base_graph_path.suffix))
                    # graphviz は .render/.pipe 呼び出しでファイルを書き出す
                    # .render を使って余分なサフィックス処理を避ける
                    # graphviz.Source.render はサフィックスを追加するので Digraph から直接出力する
                    dot.render(filename=str(Path(out_path).with_suffix('')), cleanup=True)
            except Exception as e:
                print(f'Graphviz 出力エラー: {e}', file=sys.stderr)

    def generate_detailed_adjacency(self, mappers: List[MapperEntity], output_path: str, graph_path: Optional[str] = None) -> None:
        """Javaファイル / XML ファイル / SQL文 の詳細な関係性を出力する

        ノード種別:
          - file: ファイル名 (SampleMapper.xml / SampleMapper.java)
          - method: Java内のメソッド/アノテーションSQL
          - stmt: XML内のSQL文 (namespace.id)
        """
    # ノードとエッジを構築
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Tuple[str, str]] = []

    # ノード追加のヘルパー
        def add_node(node_id: str, label: str, ntype: str):
            if node_id not in nodes:
                nodes[node_id] = {'label': label, 'type': ntype}

        namespace_to_file: Dict[str, str] = {}
        for m in mappers:
            fname = Path(m.file_path).name
            file_node = f'file:{fname}'
            ftype = 'xml' if fname.endswith('.xml') else 'java'
            add_node(file_node, fname, ftype)
            if m.namespace:
                namespace_to_file[m.namespace] = file_node

            # statements と includes
        for m in mappers:
            file_node = f'file:{Path(m.file_path).name}'
            # statements
            for stmt in m.statements:
                stmt_node = f'stmt:{m.namespace}.{stmt.id}' if m.namespace else f'stmt:{Path(m.file_path).name}.{stmt.id}'
                add_node(stmt_node, stmt.id, 'stmt')
                edges.append((file_node, stmt_node))

            # includes -> 断片を持つファイルに解決を試みる
            for include_ref in m.includes:
                resolved_ns = None
                if '.' in include_ref:
                    resolved_ns = include_ref.rsplit('.', 1)[0]
                else:
                    # マッパー間で断片を検索
                    for ns, mm in ((mm.namespace, mm) for mm in mappers):
                        if ns is None:
                            continue
                        qualified = f'{ns}.{include_ref}'
                        if qualified in mm.sql_fragments or include_ref in mm.sql_fragments:
                            resolved_ns = ns
                            break
                if resolved_ns and resolved_ns in namespace_to_file:
                    target_file_node = namespace_to_file[resolved_ns]
                    edges.append((file_node, target_file_node))

            # 依存関係 (resultMap の extends)
            for dep in m.dependencies:
                dep_ns = None
                if '.' in dep:
                    dep_ns = dep.rsplit('.', 1)[0]
                else:
                    # 検索
                    for mm in mappers:
                        if dep in mm.result_maps:
                            dep_ns = mm.namespace
                            break
                if dep_ns and dep_ns in namespace_to_file:
                    edges.append((file_node, namespace_to_file[dep_ns]))

    # Java のメソッドと XML の stmt のリンク: Java マッパーのメソッド id があり、同一名前空間の XML に一致する stmt id があればリンクする
    # 名前空間ごとの XML stmt id を特定
        stmts_by_ns = defaultdict(set)
        for m in mappers:
            for stmt in m.statements:
                if m.namespace:
                    stmts_by_ns[m.namespace].add(stmt.id)

        for m in mappers:
            if Path(m.file_path).suffix == '.java':
                file_node = f'file:{Path(m.file_path).name}'
                for stmt in m.statements:
                    # method_<line> のような method id や実際のメソッド名など; id を使って XML の stmt と一致させる
                    if m.namespace and stmt.id in stmts_by_ns.get(m.namespace, set()):
                        target_stmt_node = f'stmt:{m.namespace}.{stmt.id}'
                        edges.append((file_node, target_stmt_node))
                    else:
                        # standalone method node
                        method_node = f'method:{m.namespace}.{stmt.id}' if m.namespace else f'method:{Path(m.file_path).name}.{stmt.id}'
                        add_node(method_node, stmt.id, 'method')
                        edges.append((file_node, method_node))

    # 隣接マッピングを構築
        adjacency: Dict[str, List[str]] = defaultdict(list)
        reverse_adj: Dict[str, List[str]] = defaultdict(list)
        for src, dst in edges:
            adjacency[src].append(dst)
            reverse_adj[dst].append(src)

    # Markdown を書き出す
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('# 詳細マッパー隣接レポート\n')
                f.write(f'生成日時: {datetime.now().isoformat()}\n\n')
                f.write('## ノード\n')
                for nid, meta in nodes.items():
                    f.write(f'- {nid} ({meta["type"]}): {meta["label"]}\n')
                f.write('\n## エッジ\n')
                for s, d in edges:
                    f.write(f'- {s} -> {d}\n')
        except Exception as e:
            print(f'Detailed adjacency markdown 出力エラー: {e}', file=sys.stderr)

        # JSON
        try:
            json_path = str(Path(output_path).with_suffix('.detailed.json'))
            with open(json_path, 'w', encoding='utf-8') as jf:
                json.dump({'nodes': nodes, 'edges': edges, 'adjacency': adjacency, 'reverse_adjacency': reverse_adj}, jf, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f'Detailed adjacency JSON 出力エラー: {e}', file=sys.stderr)

        # Graphviz
        if graph_path:
            try:
                import graphviz
                dot = graphviz.Digraph(format='png')
                dot.attr(rankdir='LR')
                # Graphviz のポート構文問題を避けるために安全なノード ID を作成
                safe_map: Dict[str, str] = {}
                for i, nid in enumerate(nodes.keys()):
                    safe_map[nid] = f'n{i}'

                # 安全な ID を使って装飾付きノードを追加（ラベルは元の値を表示）
                for nid, meta in nodes.items():
                    safe = safe_map[nid]
                    ntype = meta['type']
                    label = meta['label']
                    if ntype == 'java':
                        dot.node(safe, label=label, shape='box', style='filled', fillcolor='lightgoldenrod')
                    elif ntype == 'xml':
                        dot.node(safe, label=label, shape='rectangle', style='filled', fillcolor='lightcyan')
                    elif ntype == 'stmt':
                        dot.node(safe, label=label, shape='note', style='filled', fillcolor='lightgrey')
                    else:
                        dot.node(safe, label=label)

                # 安全な ID を使ってエッジを追加
                for src, dst in edges:
                    s = safe_map.get(src, None)
                    d = safe_map.get(dst, None)
                    if s and d:
                        dot.edge(s, d)

                # Java ファイルが XML の名前空間内の stmt を参照する場合、明示的な Java->XML リファレンスエッジ（破線青）を追加
                for src, dst in edges:
                    # src is file node
                    if src.startswith('file:') and dst.startswith('stmt:'):
                        src_meta = nodes.get(src)
                        if src_meta and src_meta.get('type') == 'java':
                            # extract namespace from stmt id: 'stmt:namespace.id'
                            rest = dst.split(':', 1)[1]
                            ns = rest.split('.', 1)[0] if '.' in rest else None
                            if ns:
                                target_file_node = namespace_to_file.get(ns)
                                if target_file_node:
                                    s_safe = safe_map.get(src)
                                    t_safe = safe_map.get(target_file_node)
                                    if s_safe and t_safe:
                                        dot.edge(s_safe, t_safe, color='blue', style='dashed')

                base = str(Path(graph_path).with_suffix(''))
                dot.render(filename=base, cleanup=True)
            except Exception as e:
                print(f'Detailed Graphviz 出力エラー: {e}', file=sys.stderr)
    
    def _add_table_list(self, sql: Dict) -> None:
        """テーブル一覧の追加"""
        section = f"""## テーブル情報

**アクセステーブル数**: {sql.get('table_count', 0)}

### テーブル一覧"""
        
        tables = sql.get('table_list', [])
        if tables:
            # 3列で表示
            for i in range(0, len(tables), 3):
                row = tables[i:i+3]
                section += "\n- " + ", ".join(f"`{t}`" for t in row)
        else:
            section += "\n*テーブルが検出されませんでした*"
        
        self.report_sections.append(section)

        # テーブルごとの CRUD 件数（存在すれば表示）
        table_crud = sql.get('table_crud')
        if table_crud:
            crud_section = "\n\n### テーブル別 CRUD 件数\n\n| テーブル | SELECT | INSERT | UPDATE | DELETE |\n|---|---:|---:|---:|---:|"
            for table, counts in sorted(table_crud.items(), key=lambda x: x[0]):
                s = counts.get('SELECT', 0)
                i = counts.get('INSERT', 0)
                u = counts.get('UPDATE', 0)
                d = counts.get('DELETE', 0)
                crud_section += f"\n| `{table}` | {s} | {i} | {u} | {d} |"
            self.report_sections.append(crud_section)
    
    def _add_mapper_details(self, mappers: List[MapperEntity]) -> None:
        """Mapper詳細の追加"""
        details = f"""## Mapperファイル一覧

| ファイル名 | 名前空間 | SQL文数 | パラメータ数 |
|-----------|---------|--------|------------|"""
        
        # ファイル名でソート
        sorted_mappers = sorted(mappers, key=lambda m: Path(m.file_path).name)
        
        for mapper in sorted_mappers:
            file_name = Path(mapper.file_path).name
            namespace_short = mapper.namespace.split('.')[-1] if mapper.namespace else '-'
            details += f"\n| {file_name} | {namespace_short} | {len(mapper.statements)} | {len(mapper.parameters)} |"
        
        self.report_sections.append(details)


class MapperAnalyzer:
    """メインコントローラークラス"""

    def __init__(self, root_path: str, output_path: str = "mapper_analysis.md", csv_output: Optional[str] = None, table_map: Optional[Dict[str, str]] = None):
        self.root_path = Path(root_path)
        self.output_path = output_path
        self.csv_output = csv_output
        # optional mapping from short table names to full names
        self.table_map = table_map
        # adjacency output paths (optional)
        self.adjacency_output = None
        self.graph_output = None
        self.scanner = DirectoryScanner()
        self.parser = MapperFileParser()
        self.sql_analyzer = SQLStatementAnalyzer()
        self.dependency_graph = DependencyGraph()
        self.metrics_calculator = MetricsCalculator()
        self.report_generator = ReportGenerator()
    
    def analyze(self) -> None:
        """完全な分析プロセスの実行"""
        print("MyBatis Mapper分析を開始...")
        
        # 1. ファイルスキャン
        print("Mapperファイルをスキャン中...")
        mapper_files = self.scanner.scan(self.root_path)
        print(f"  {len(mapper_files)}個のファイルを発見")
        
        # 2. ファイル解析
        print("ファイルを解析中...")
        mappers = []
        for file_path in mapper_files:
            mapper = self.parser.parse(file_path)
            if mapper:
                mappers.append(mapper)
        print(f"  {len(mappers)}個のMapperを解析完了")
        
        if not mappers:
            print("警告: 解析可能なMapperファイルが見つかりませんでした")
            return
        
        # 3. SQL分析
        print("SQL文を分析中...")
        sql_analysis = self.sql_analyzer.analyze(mappers)
        
        # 4. 依存関係構築
        print("依存関係グラフを構築中...")
        self.dependency_graph.build(mappers)
        dependency_metrics = self.dependency_graph.get_metrics()
        
        # 5. メトリクス計算
        print("メトリクスを計算中...")
        general_metrics = self.metrics_calculator.calculate(mappers)
        
        # 5.5: SQLごとのメトリクスCSV出力（指定されていれば）
        if getattr(self, 'csv_output', None):
            try:
                csv_path = str(self.csv_output)
                print(f"SQLメトリクスCSVを出力: {csv_path}")
                self.metrics_calculator.export_sql_metrics_csv(mappers, csv_path, table_map=getattr(self, 'table_map', None))
            except Exception as e:
                print(f"CSV出力エラー: {e}", file=sys.stderr)
        
        # 6. レポート生成
        print("レポートを生成中...")
        self.report_generator.generate(
            mappers,
            sql_analysis,
            dependency_metrics,
            general_metrics,
            self.output_path
        )

        # 隣接レポート（別ファイル）出力
        if getattr(self, 'adjacency_output', None):
            adj_path = str(self.adjacency_output)
            graph_path = getattr(self, 'graph_output', None)
            print(f"隣接レポートを出力: {adj_path}")
            try:
                self.report_generator.generate_adjacency_report(dependency_metrics, adj_path, graph_path)
            except Exception as e:
                print(f"隣接レポート出力エラー: {e}", file=sys.stderr)

        # 詳細隣接（ファイル/メソッド/SQL文）出力
        if getattr(self, 'adjacency_output', None):
            try:
                if self.adjacency_output:
                    detailed_path = str(Path(self.adjacency_output).with_name(Path(self.adjacency_output).stem + '_detailed.md'))
                else:
                    detailed_path = None

                detailed_graph = None
                if getattr(self, 'graph_output', None):
                    if self.graph_output:
                        detailed_graph = str(Path(self.graph_output).with_name(Path(self.graph_output).stem + '_detailed').with_suffix(Path(self.graph_output).suffix))

                if detailed_path:
                    print(f"詳細隣接レポートを出力: {detailed_path}")
                    self.report_generator.generate_detailed_adjacency(mappers, detailed_path, detailed_graph)
            except Exception as e:
                print(f"詳細隣接レポート出力エラー: {e}", file=sys.stderr)

        print(f"分析完了: {self.output_path}")


def main():
    """エントリーポイント"""
    parser = argparse.ArgumentParser(
        description='MyBatis Mapperディレクトリを解析し、Markdownレポートを生成'
    )
    parser.add_argument(
        'path',
        help='解析対象のルートディレクトリパス'
    )
    parser.add_argument(
        '-o', '--output',
    default='output/mapper_analysis.md',
    help='出力レポートファイル名 (デフォルト: output/mapper_analysis.md)'
    )
    parser.add_argument(
        '--csv',
        dest='csv',
        default='output/mapper_sql_metrics.csv',
        help='SQLごとのメトリクスを出力するCSVファイルパス (デフォルト: output/mapper_sql_metrics.csv)。無効化する場合は空文字を指定してください。'
    )
    parser.add_argument(
        '--adjacency',
        dest='adjacency',
    default='',
        help='隣接リスト（Mapper呼び出し関係）を出力するMarkdownファイルパス (デフォルト: output/mapper_adjacency.md)。無効化する場合は空文字を指定してください。'
    )
    parser.add_argument(
        '--graph',
        dest='graph',
    default='',
        help='依存関係グラフを出力する画像ファイルパス (デフォルト: output/mapper_graph.png)。無効化する場合は空文字を指定してください。'
    )
    parser.add_argument(
        '--table-map',
        dest='table_map',
        default='',
        help='テーブル略称から正式名へのマッピングを定義したJSONファイルのパス（例: {"TBL": "TABLE_FULL_NAME"}）。指定しない場合はマッピングを適用しません。'
    )
    
    args = parser.parse_args()
    
    # パスの検証
    if not Path(args.path).exists():
        print(f"エラー: パス '{args.path}' が存在しない", file=sys.stderr)
        sys.exit(1)
    
    if not Path(args.path).is_dir():
        print(f"エラー: '{args.path}' はディレクトリではない", file=sys.stderr)
        sys.exit(1)
    
    # 分析実行
    # Ensure output directory exists (default is ./output/...)
    out_path = Path(args.output)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    analyzer = MapperAnalyzer(args.path, str(out_path), csv_output=args.csv)
    # adjacency and graph outputs
    analyzer.adjacency_output = args.adjacency if args.adjacency else None
    analyzer.graph_output = args.graph if args.graph else None
    # load table map if provided
    table_map = None
    if getattr(args, 'table_map', None):
        try:
            tm_path = Path(args.table_map)
            if tm_path.exists():
                with open(tm_path, 'r', encoding='utf-8') as tf:
                    table_map = json.load(tf)
        except Exception as e:
            print(f"テーブルマップ読み込みエラー: {e}", file=sys.stderr)
    analyzer.table_map = table_map
    analyzer.analyze()


if __name__ == '__main__':
    main()
