import pytest

# Apply module-level marks
pytestmark = pytest.mark.integration


@pytest.mark.parametrize("obj_type", ['event', 'superevent'])
def test_logs(client, create_obj, obj_type):
    # Create event or superevent
    obj, obj_id = create_obj(obj_type)

    # Create a log
    comment = 'test comment'
    response = client.write_log(obj_id, comment)
    assert response.status_code == 201
    data = response.json()
    assert data['comment'] == comment
    log_N = data['N']

    # Pull down list of logs for the event
    response = client.logs(obj_id)
    assert response.status_code == 200
    data = response.json()
    assert len(data['log']) >= 1

    # Pull down individual log
    response = client.logs(obj_id, log_N)
    assert response.status_code == 200
    data = response.json()
    assert data['comment'] == comment
    assert data['N'] == log_N


@pytest.mark.parametrize("obj_type", ['event', 'superevent'])
def test_log_creation_with_tags(client, create_obj, obj_type):
    # Create event or superevent
    obj, obj_id = create_obj(obj_type)

    # Create a log with tags
    comment = 'test log with tags'
    tags = ['test_tag1', 'test_tag2']
    response = client.write_log(obj_id, comment, tag_name=tags)
    assert response.status_code == 201
    data = response.json()
    assert data['comment'] == comment


@pytest.mark.parametrize("obj_type", ['event', 'superevent'])
def test_log_creation_with_label(client, create_obj, obj_type):
    # Create event or superevent
    obj, obj_id = create_obj(obj_type)

    # Create a log with tags
    comment = 'test log with label'
    label = 'DQV'
    response = client.write_log(obj_id, comment, label=label)
    assert response.status_code == 201
    data = response.json()
    assert data['comment'] == comment

    # get the list of labels:
    response = client.labels(obj_id)
    labels_list = [lab['name'] for lab in response.json()['labels']]
    assert label in labels_list


@pytest.mark.parametrize("obj_type", ['event', 'superevent'])
def test_log_tag_and_untag(client, create_obj, obj_type):
    # Create event or superevent
    obj, obj_id = create_obj(obj_type)

    # Create a log
    comment = 'test log, add/remove tags later'
    response = client.write_log(obj_id, comment)
    assert response.status_code == 201
    data = response.json()
    assert data['comment'] == comment
    log_N = data['N']

    # Add a tag
    tag = 'test_tag'
    response = client.add_tag(obj_id, log_N, tag)
    assert response.status_code == 201

    # Get log and check tag status
    response = client.logs(obj_id, log_N)
    assert response.status_code == 200
    data = response.json()
    assert len(data['tag_names']) == 1
    assert tag in data['tag_names']

    # Remove tag
    response = client.remove_tag(obj_id, log_N, tag)
    assert response.status_code == 204

    # Get log and check tag status
    response = client.logs(obj_id, log_N)
    assert response.status_code == 200
    data = response.json()
    assert len(data['tag_names']) == 0
