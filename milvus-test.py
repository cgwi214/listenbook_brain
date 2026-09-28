from utils.milvus_util import get_milvus_client


client = get_milvus_client()


res = client.query(
    collection_name="kb_graph_entity_names_v2",
    filter="item_name like '%RS PRO%'",
    output_fields=[
        "entity_name",
        "item_name",
        "context"
    ],
    limit=10
)

print(res)